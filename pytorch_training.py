from multiprocessing import Process
from pathlib import Path
from time import perf_counter

import click
import torch
from torch import manual_seed
from tqdm import tqdm

from utils import log_jetson_stats, get_data_loaders, output_label, get_pred, count_parameters


def get_corrects(logit, target):
    ''' Obtain number of corrects predictions '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    return corrects


@click.command()
@click.option('--model-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--data-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--device', type=click.Choice(['cpu', 'cuda']), default='cpu')
@click.option('--epochs', type=int, default=5)
@click.option('--batch-size', type=int, default=8)
@click.option("--profile", is_flag=True)
@click.option("--evaluate", is_flag=True)
@click.option('--seed', type=int, default=42)
def training(model_path: Path, data_path: Path, device: str, epochs: int, batch_size: int, profile: bool, evaluate: bool, seed: int):
    manual_seed(seed)
    if profile:
        platform_stats_process = Process(target=log_jetson_stats, args=('onnxruntime', model_path.name, data_path.name, device,))
        platform_stats_process.start()

    print('Creating dataloaders...')
    train_loader, test_loader = get_data_loaders(data_path, batch_size)

    print(f'Creating model with {device} device...')
    model = torch.load(model_path)
    if device == 'cuda':
        device = torch.device('cuda')
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    print(f'Model trainable parameters: {count_parameters(model)}')

    print('Setting criterion...')
    criterion = torch.nn.CrossEntropyLoss()

    print('Creating optimizer...')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    training_time = 0.
    evaluate_time = 0.

    print(f'Starting training loop for {epochs} epochs...')
    for epoch in range(epochs):
        train_running_loss = 0.0
        test_running_loss = 0.0

        train_corrects = []
        test_corrects = []

        model = model.train()
        train_start = perf_counter()
        ## training step
        for i, (images, labels) in enumerate(tqdm(train_loader)):

            images = images.to(device)
            labels = labels.to(device)

            ## forward + backprop + loss
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            ## update model params
            optimizer.step()

            train_running_loss += loss.detach().item()
            train_corrects.append(get_corrects(logits, labels))

        train_time = perf_counter() - train_start
        training_time += train_time

        model.eval()
        eval_start = perf_counter()
        with torch.no_grad():
            for k, (images, labels) in enumerate(tqdm(test_loader)):
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                test_running_loss += loss.detach().item()
                test_corrects.append(get_corrects(logits, labels))

        eval_time = perf_counter() - eval_start
        evaluate_time += eval_time
        epoch_time = train_time + eval_time

        train_acc = sum(train_corrects) / 50000
        test_acc = sum(test_corrects) / 10000

        print(
            f'Epoch: {epoch+1} |',
            f' Loss: {train_running_loss/i:.4f} | Train Accuracy: {train_acc:.4f} |',
            f' Test loss: {test_running_loss/k:.4f} | Test Accuracy: {test_acc:.4f} |',
            f' Epoch time: {epoch_time:.2f}s')

    if profile:
        print('Terminating platform stats logger...')
        platform_stats_process.terminate()

    print(f'Training completed in {training_time+evaluate_time:.2f}s')
    print(f'Average training time per epoch: {training_time / epochs:.2f}s')
    print(f'Average eval time per epoch: {evaluate_time / epochs:.2f}s')

    if evaluate:
        print('Inferencing trained model...')
        # Testing model with one example from test set
        data, label = next(iter(test_loader))

        idx = 0
        data, label = data[idx], label.numpy()[idx]

        output = model(data.unsqueeze(0).to(device))

        print('Predicted Label : ', output_label(get_pred(output.cpu().detach().numpy())[0], data_path.name))
        print('GT label: ', output_label(label, data_path.name))


if __name__ == '__main__':
    training()
