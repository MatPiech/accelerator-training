import logging
from pathlib import Path

import click
import numpy as np
import onnx
from onnx import numpy_helper
import torch

from accelerator_training.utils import setup_logging

setup_logging(logging.INFO)
logger = logging.getLogger("change_model_output_features")


def _find_last_linear_module(model: torch.nn.Module) -> tuple[str, torch.nn.Linear]:
    last_name = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            last_name = name
            last_module = module
    if last_name is None or last_module is None:
        raise ValueError("No torch.nn.Linear layer found in the model.")
    return last_name, last_module


def _set_module_by_name(model: torch.nn.Module, name: str, new_module: torch.nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _resize_linear_torch(model_path: Path, output_features: int, output_path: Path) -> None:
    logger.info("Loading PyTorch model...")
    model = torch.load(model_path, weights_only=False, map_location="cpu")

    layer_name, linear = _find_last_linear_module(model)
    in_features = linear.in_features
    out_features_old = linear.out_features

    if out_features_old == output_features:
        logger.info("Output features already match. Saving without changes.")
        torch.save(model, output_path)
        return

    new_linear = torch.nn.Linear(in_features, output_features, bias=linear.bias is not None)

    with torch.no_grad():
        new_linear.weight.zero_()
        if new_linear.bias is not None:
            new_linear.bias.zero_()
        rows = min(out_features_old, output_features)
        new_linear.weight[:rows].copy_(linear.weight[:rows])
        if new_linear.bias is not None and linear.bias is not None:
            new_linear.bias[:rows].copy_(linear.bias[:rows])

    _set_module_by_name(model, layer_name, new_linear)
    logger.info("Saving updated PyTorch model...")
    torch.save(model, output_path)


def _initializer_by_name(graph: onnx.GraphProto, name: str) -> onnx.TensorProto | None:
    for init in graph.initializer:
        if init.name == name:
            return init
    return None


def _replace_initializer(graph: onnx.GraphProto, name: str, array: np.ndarray) -> None:
    new_init = numpy_helper.from_array(array, name=name)
    for init in graph.initializer:
        if init.name == name:
            graph.initializer.remove(init)
            graph.initializer.append(new_init)
            return
    graph.initializer.append(new_init)


def _update_value_info_shape(graph: onnx.GraphProto, tensor_name: str, new_out_features: int) -> None:
    def _update_value(value_info: onnx.ValueInfoProto) -> bool:
        if value_info.name != tensor_name:
            return False
        shape = value_info.type.tensor_type.shape
        if len(shape.dim) >= 2:
            shape.dim[-1].dim_value = new_out_features
        return True

    for output in graph.output:
        if _update_value(output):
            return
    for value_info in graph.value_info:
        if _update_value(value_info):
            return


def _find_last_linear_onnx(graph: onnx.GraphProto):
    output_to_node = {}
    for node in graph.node:
        for output in node.output:
            output_to_node[output] = node

    for node in reversed(graph.node):
        if node.op_type == "Gemm":
            return ("gemm", node, None)

    for node in reversed(graph.node):
        if node.op_type != "Add":
            continue
        matmul_node = None
        bias_name = None
        for input_name in node.input:
            if input_name in output_to_node and output_to_node[input_name].op_type == "MatMul":
                matmul_node = output_to_node[input_name]
            else:
                bias_name = input_name
        if matmul_node is None or bias_name is None:
            continue
        weight_name = matmul_node.input[1] if len(matmul_node.input) > 1 else None
        if weight_name is None:
            continue
        return ("matmul_add", matmul_node, bias_name)

    raise ValueError("Could not find a supported last linear layer (Gemm or MatMul+Add).")


def _resize_linear_onnx(model_path: Path, output_features: int, output_path: Path) -> None:
    logger.info("Loading ONNX model...")
    model = onnx.load(model_path)
    graph = model.graph

    kind, node, bias_name = _find_last_linear_onnx(graph)

    if kind == "gemm":
        weight_name = node.input[1]
        bias_name = node.input[2] if len(node.input) > 2 else None
        trans_b = 0
        for attr in node.attribute:
            if attr.name == "transB":
                trans_b = attr.i
                break
    else:
        weight_name = node.input[1]
        trans_b = 0

    weight_init = _initializer_by_name(graph, weight_name)
    if weight_init is None:
        raise ValueError(f"Weight initializer {weight_name} not found in ONNX graph.")

    weight = numpy_helper.to_array(weight_init)

    if kind == "gemm" and trans_b == 1:
        old_out, in_features = weight.shape
        new_shape = (output_features, in_features)
        copy_rows = min(old_out, output_features)
        new_weight = np.random.normal(0.0, 0.02, size=new_shape).astype(weight.dtype)
        new_weight[:copy_rows] = weight[:copy_rows]
    else:
        in_features, old_out = weight.shape
        new_shape = (in_features, output_features)
        copy_cols = min(old_out, output_features)
        new_weight = np.random.normal(0.0, 0.02, size=new_shape).astype(weight.dtype)
        new_weight[:, :copy_cols] = weight[:, :copy_cols]

    _replace_initializer(graph, weight_name, new_weight)

    if bias_name is not None:
        bias_init = _initializer_by_name(graph, bias_name)
        if bias_init is not None:
            bias = numpy_helper.to_array(bias_init)
            new_bias = np.zeros((output_features,), dtype=bias.dtype)
            copy_bias = min(bias.shape[0], output_features)
            new_bias[:copy_bias] = bias[:copy_bias]
            _replace_initializer(graph, bias_name, new_bias)

    output_tensor_name = node.output[0] if node.output else None
    if output_tensor_name:
        _update_value_info_shape(graph, output_tensor_name, output_features)

    logger.info("Saving updated ONNX model...")
    onnx.save(model, output_path)


def _default_output_path(model_path: Path, output_features: int) -> Path:
    return model_path.with_name(f"{model_path.stem}_out{output_features}{model_path.suffix}")


@click.command()
@click.option("--model-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-features", type=int, required=True)
@click.option("--output-path", type=click.Path(path_type=Path))
def main(model_path: Path, output_features: int, output_path: Path | None):
    if output_features <= 0:
        raise click.BadParameter("output-features must be a positive integer.")

    if output_path is None:
        output_path = _default_output_path(model_path, output_features)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = model_path.suffix.lower()
    if suffix in {".pth", ".pt"}:
        _resize_linear_torch(model_path, output_features, output_path)
    elif suffix == ".onnx":
        _resize_linear_onnx(model_path, output_features, output_path)
    else:
        raise click.BadParameter(f"Unsupported model extension: {suffix}")

    logger.info(f"Done. Updated model saved to: {output_path}")


if __name__ == "__main__":
    main()
