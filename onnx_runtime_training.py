from multiprocessing import Process
from pathlib import Path
from time import perf_counter

import click
import onnx
import numpy as np
from onnxruntime import InferenceSession
from onnxruntime.training.api import CheckpointState, Module, Optimizer
from tqdm import tqdm

from utils import log_jetson_stats, get_data_loaders, output_label, get_pred


def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (get_pred(logit) == target.numpy()).sum()
    accuracy = corrects / batch_size
    return accuracy.item()


@click.command()
@click.option('--model-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--data-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--device', type=click.Choice(['cpu', 'cuda', 'hailo']), default='cpu')
@click.option('--epochs', type=int, default=5)
@click.option('--batch-size', type=int, default=64)
@click.option("--profile", is_flag=True)
def training(model_path: Path, data_path: Path, device: str, epochs: int, batch_size: int, profile: bool):
    if profile:
        platform_stats_process = Process(target=log_jetson_stats, args=('onnxruntime', model_path.name, data_path.name, device,))
        platform_stats_process.start()

    print('Creating dataloaders...')
    train_loader, test_loader = get_data_loaders(data_path, batch_size)

    # artifacts path
    artifacts_path = Path(model_path)

    print('Loading artifacts...')
    # Create checkpoint state
    state = CheckpointState.load_checkpoint(artifacts_path / 'checkpoint')

    # Create module
    print(f'Creating model with {device} device...')
    model = Module(
        artifacts_path / 'training_model.onnx',
        state,
        artifacts_path / 'eval_model.onnx',
        device=device,
    )

    # Create optimizer
    optimizer = Optimizer(artifacts_path / 'optimizer_model.onnx', model)
    optimizer.set_learning_rate(0.001)

    print(f'Starting training loop for {epochs} epochs...')
    for epoch in range(epochs):
        epoch_start = perf_counter()
        # Training Loop
        model.train()
        train_acc = []
        train_losses = []

        for _, (data, target) in enumerate(tqdm(train_loader)):
            forward_inputs = [data.numpy(), target.numpy().astype(np.int64)]
            train_loss, logits = model(*forward_inputs)
            train_acc.append(get_accuracy(logits, target, batch_size))
            optimizer.step()
            model.lazy_reset_grad()
            train_losses.append(train_loss)
        
        # Test Loop
        model.eval()
        test_acc = []
        test_losses = []

        for _, (data, target) in enumerate(tqdm(test_loader)):
            forward_inputs = [data.numpy(), target.numpy().astype(np.int64)]
            test_loss, logits = model(*forward_inputs)
            test_acc.append(get_accuracy(logits, target, batch_size))
            test_losses.append(test_loss)

        epoch_time = perf_counter() - epoch_start

        print(
            f'Epoch: {epoch+1} |',
            f' Loss: {sum(train_losses) / len(train_losses):.4f} | Train Accuracy: {sum(train_acc) / len(train_acc):.2f} |',
            f' Test loss: {sum(test_losses) / len(test_losses):.4f} | Test Accuracy: {sum(test_acc) / len(test_acc):.2f} |',
            f' Epoch time: {epoch_time:.2f}s')

    if profile:
        print('Terminating platform stats logger...')
        platform_stats_process.terminate()

    print('Exporting trained model for inference...')
    m_ = onnx.load_model(artifacts_path / 'eval_model.onnx')
    output_names = [o_.name for o_ in m_.graph.output]
    model.export_model_for_inferencing(artifacts_path / 'inference_model.onnx', output_names[1:])

    print('Inferencing trained model...')
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

    print('Predicted Label : ', output_label(get_pred(output[0])[0], data_path.name))
    print('GT label: ', output_label(label, data_path.name))


if __name__ == '__main__':
    training()
