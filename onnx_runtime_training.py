import logging
from pathlib import Path
from time import perf_counter

import click
import onnx
import numpy as np
from onnxruntime import InferenceSession, set_seed
from onnxruntime.training.api import CheckpointState, Module, Optimizer
from torch import manual_seed
from tqdm import tqdm

from utils import get_data_loaders, output_label, get_pred, setup_logging


# configure logging with project-wide format
setup_logging()


def get_corrects(logit, target):
    ''' Obtain number of corrects predictions '''
    corrects = (get_pred(logit) == target.numpy()).sum()
    return corrects


@click.command()
@click.option('--model-dir', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--data-path', type=click.Path(path_type=Path), required=True)
@click.option('--device', type=click.Choice(['cpu', 'cuda', 'hailo']), default='cpu')
@click.option('--epochs', type=int, default=1)
@click.option('--batch-size', type=int, default=1)
@click.option("--train", is_flag=True)
@click.option("--inference-sample", is_flag=True)
@click.option('--seed', type=int, default=42)
def training(model_dir: Path, data_path: Path, device: str, epochs: int, batch_size: int, train: bool, inference_sample: bool, seed: int):
    manual_seed(seed)
    set_seed(seed)

    logging.info('Creating dataloaders...')
    train_loader, test_loader = get_data_loaders(data_path, batch_size)
    train_len = train_loader.__len__() * batch_size
    test_len = test_loader.__len__()

    # artifacts path
    artifacts_path = Path(model_dir)

    logging.info('Loading artifacts...')
    # Create checkpoint state
    state = CheckpointState.load_checkpoint(artifacts_path / 'checkpoint')

    # Create module
    logging.info(f'Creating model with {device} device...')
    model = Module(
        artifacts_path / 'training_model.onnx',
        state,
        artifacts_path / 'eval_model.onnx',
        device=device,
    )

    logging.info(f'Model parameters: {model.get_parameters_size(trainable_only=False)}')

    if train:
        logging.info(f'Model trainable parameters: {model.get_parameters_size(trainable_only=True)}')

        # Create optimizer
        optimizer = Optimizer(artifacts_path / 'optimizer_model.onnx', model)
        optimizer.set_learning_rate(0.001)

        training_time = 0.

        logging.info(f'Starting training loop for {epochs} epochs...')
        for epoch in range(epochs):
            train_corrects, train_losses = [], []

            # Training Loop
            model.train()
            train_start = perf_counter()

            for _, (data, target) in enumerate(tqdm(train_loader)):
                forward_inputs = [data.numpy(), target.numpy().astype(np.int64)]
                train_loss, logits = model(*forward_inputs)
                train_corrects.append(get_corrects(logits, target))
                optimizer.step()
                model.lazy_reset_grad()
                train_losses.append(train_loss)

            train_time = perf_counter() - train_start
            training_time += train_time

            train_acc = sum(train_corrects) / train_len

            logging.info(' '.join([
                f'Epoch: {epoch+1} |',
                f'Loss: {sum(train_losses) / len(train_losses):.4f} | Train Accuracy: {train_acc:.4f} |',
                f'Epoch time: {train_time:.2f}s'
            ]))

        logging.info(f'Training completed in {training_time:.4f}s')
        logging.info(f'Average training time per epoch: {training_time / epochs:.4f}s')
        logging.info(f'Average training time per sample: {training_time / epochs / train_len:.4f}s')

    # Test Loop
    model.eval()
    eval_start = perf_counter()

    test_corrects, test_losses = [], []
    for _, (data, target) in enumerate(tqdm(test_loader)):
        forward_inputs = [data.numpy(), target.numpy().astype(np.int64)]
        test_loss, logits = model(*forward_inputs)
        test_corrects.append(get_corrects(logits, target))
        test_losses.append(test_loss)

    evaluate_time = perf_counter() - eval_start

    test_acc = sum(test_corrects) / test_len

    logging.info(f'Test loss: {sum(test_losses) / len(test_losses):.4f} | Test Accuracy: {test_acc:.4f}')
    logging.info(f'Test time: {evaluate_time:.4f}s')
    logging.info(f'Average test time per sample: {evaluate_time / test_len:.4f}s')

    if inference_sample:
        logging.info('Exporting trained model for inference...')
        m_ = onnx.load_model(artifacts_path / 'eval_model.onnx')
        output_names = [o_.name for o_ in m_.graph.output]
        model.export_model_for_inferencing(artifacts_path / 'inference_model.onnx', output_names[1:])

        logging.info('Inferencing trained model...')
        if device == 'hailo':
            ep = ['HailoExecutionProvider']
        elif device == 'cuda':
            ep = ['CUDAExecutionProvider']
        else:
            ep = ['CPUExecutionProvider']
        session = InferenceSession(artifacts_path / 'inference_model.onnx', providers=ep)

        # Testing model with one example from test set
        data, label = next(iter(test_loader))

        idx = 0
        data, label = data[idx], label.numpy()[idx]

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name 
        output = session.run([output_name], {input_name: data.numpy()[np.newaxis]})

        logging.info(f'Predicted Label : {output_label(get_pred(output[0])[0], data_path.name)}')
        logging.info(f'GT label: {output_label(label, data_path.name)}')


if __name__ == '__main__':
    training()
