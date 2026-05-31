import contextlib
from pathlib import Path

import click
import onnx
from onnx import numpy_helper


def profile_onnx_model(path: str, skip_last_layers: int = 0) -> tuple[int, int, list[int] | None]:
    """
    Load an ONNX model from `path`, infer shapes, and return (MACs, num_parameters, last_output_shape).
    MACs is an estimate for common ops (Conv, ConvTranspose, MatMul, Gemm).
    Unknown dims are treated as 1.
    """
    model = onnx.load(path)
    with contextlib.suppress(Exception):
        model = onnx.shape_inference.infer_shapes(model)

    # build initializer map and count params
    init_map = {init.name: init for init in model.graph.initializer}
    nodes = list(model.graph.node)
    if skip_last_layers > 0:
        nodes = nodes[:-skip_last_layers] if skip_last_layers < len(nodes) else []

    if skip_last_layers > 0:
        used_inits = set()
        for node in nodes:
            for name in node.input:
                if name in init_map:
                    used_inits.add(name)
        num_params = 0
        for name in used_inits:
            arr = numpy_helper.to_array(init_map[name])
            num_params += int(arr.size)
    else:
        num_params = 0
        for init in model.graph.initializer:
            arr = numpy_helper.to_array(init)
            num_params += int(arr.size)

    # build value name -> shape map from inputs, outputs, value_info
    def extract_shape(vi):
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            return []
        dims = []
        for d in tt.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            else:
                # unknown or symbolic -> treat as 1
                dims.append(1)
        return dims

    shape_map = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if vi.name not in shape_map:
            s = extract_shape(vi)
            if s:
                shape_map[vi.name] = s

    def get_shape(name):
        return shape_map.get(name, [])

    last_output_shape = None
    if nodes:
        last_output_name = nodes[-1].output[0] if nodes[-1].output else None
        if last_output_name:
            shape = get_shape(last_output_name)
            last_output_shape = shape if shape else None

    macs = 0

    for node in nodes:
        op = node.op_type
        if op in ("Conv", "ConvTranspose"):
            # inputs: [X, W, (B)]
            if len(node.input) < 2:
                continue
            w_name = node.input[1]
            out_name = node.output[0] if node.output else None

            w_init = init_map.get(w_name)
            w_shape = get_shape(w_name) if w_init is None else list(numpy_helper.to_array(w_init).shape)

            out_shape = get_shape(out_name) if out_name else []
            # typical conv weight shape: (out_channels, in_channels/groups, kH, kW)
            if len(w_shape) >= 4 and len(out_shape) >= 4:
                out_channels = int(w_shape[0])
                in_per_group = int(w_shape[1])
                k_h, k_w = int(w_shape[-2]), int(w_shape[-1])
                batch = int(out_shape[0]) if len(out_shape) >= 1 else 1
                out_h = int(out_shape[-2])
                out_w = int(out_shape[-1])
                ops_per_element = in_per_group * k_h * k_w
                macs += int(batch * out_channels * out_h * out_w * ops_per_element)
            else:
                # fallback: try to infer from shapes
                continue

        elif op in ("MatMul", "Gemm"):
            # handle simple 2D matmul: A (M,K) * B (K,N) -> M*N*K
            a_name = node.input[0] if node.input else None
            b_name = node.input[1] if len(node.input) > 1 else None
            a_shape = get_shape(a_name) if a_name else []
            b_shape = get_shape(b_name) if b_name else []
            if op == "Gemm":
                # Gemm may include transposes via attributes; handle common case without transposes
                trans_a = 0
                trans_b = 0
                for attr in node.attribute:
                    if attr.name == "transA":
                        trans_a = int(attr.i)
                    if attr.name == "transB":
                        trans_b = int(attr.i)
                if trans_a:
                    a_shape = list(reversed(a_shape))
                if trans_b:
                    b_shape = list(reversed(b_shape))
            if len(a_shape) >= 2 and len(b_shape) >= 2:
                m = a_shape[-2]
                k = a_shape[-1]
                n = b_shape[-1]
                batch = 1
                # handle batched matmul (leading dims)
                if len(a_shape) > 2 or len(b_shape) > 2:
                    # approximate using first dim as batch
                    batch = int(a_shape[0]) if len(a_shape) > 2 else int(b_shape[0])
                macs += int(batch * m * n * k)
            else:
                continue
        else:
            # skip other ops for now
            continue

    return int(macs), int(num_params), last_output_shape


@click.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option(
    "--skip-last-layers",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Skip the last N nodes when calculating MACs and params.",
)
def main(model_path: str, skip_last_layers: int) -> None:
    macs, num_params, last_output_shape = profile_onnx_model(model_path, skip_last_layers=skip_last_layers)
    gmacs = macs / 1e9
    mparams = num_params / 1e6
    model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"MACs: {gmacs:.3f} G")
    print(f"Params: {mparams:.3f} M")
    print(f"Model size: {model_size_mb:.2f} MB")
    if last_output_shape is None:
        print("Last output shape: unknown")
    else:
        print(f"Last output shape: {last_output_shape}")


if __name__ == "__main__":
    main()
