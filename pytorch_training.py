import logging
from pathlib import Path
from time import perf_counter

import click
import torch
from torch import manual_seed
from tqdm import tqdm

from utils import get_data_loaders, output_label, get_pred, count_parameters


logging.basicConfig(level=logging.INFO)
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger('pytorch_training')
logger.setLevel(logging.INFO)


def get_corrects(logit, target):
    ''' Obtain number of corrects predictions '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    return corrects


@click.command()
@click.option('--model-path', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--data-path', type=click.Path(path_type=Path), required=True)
@click.option('--device', type=click.Choice(['cpu', 'cuda']), default='cpu')
@click.option('--epochs', type=int, default=1)
@click.option('--batch-size', type=int, default=1)
@click.option("--train", is_flag=True)
@click.option("--inference-sample", is_flag=True)
@click.option('--seed', type=int, default=42)
def training(model_path: Path, data_path: Path, device: str, epochs: int, batch_size: int, train: bool, inference_sample: bool, seed: int):
    manual_seed(seed)

    logger.info('Creating dataloaders...')
    train_loader, test_loader = get_data_loaders(data_path, batch_size)
    train_len = train_loader.__len__() * batch_size
    test_len = test_loader.__len__()

    logger.info(f'Creating model with {device} device...')
    model = torch.load(model_path, weights_only=False)
    if device == 'cuda':
        device = torch.device('cuda')
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    logger.info(f'Model trainable parameters: {count_parameters(model)}')

    logger.info('Setting criterion...')
    criterion = torch.nn.CrossEntropyLoss()

    logger.info('Creating optimizer...')
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    training_time = 0.
    evaluate_time = 0.

    if train:
        logger.info(f'Starting training loop for {epochs} epochs...')
        for epoch in range(epochs):
            train_running_loss = 0.0

            train_corrects = []

            model = model.train()
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d | torch.nn.LayerNorm | torch.nn.GroupNorm | torch.nn.InstanceNorm2d):
                    m.eval()
                    m.track_running_stats = False

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

            train_acc = sum(train_corrects) / train_len

            logger.info(' '.join([
                f'Epoch: {epoch+1} |',
                f'Loss: {train_running_loss/i:.4f} | Train Accuracy: {train_acc:.4f} |',
                f'Epoch time: {train_time:.2f}s'
            ]))

        logger.info(f'Training completed in {training_time+evaluate_time:.4f}s')
        logger.info(f'Average training time per epoch: {training_time / epochs:.4f}s')
        logger.info(f'Average training time per sample: {training_time / epochs / train_len:.4f}s')

    model.eval()

    test_running_loss = 0.0
    test_corrects = []

    eval_start = perf_counter()
    with torch.no_grad():
        for k, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            test_running_loss += loss.detach().item()
            test_corrects.append(get_corrects(logits, labels))

    evaluate_time = perf_counter() - eval_start

    test_acc = sum(test_corrects) / test_len

    logger.info(f'Test loss: {test_running_loss/k:.4f} | Test Accuracy: {test_acc:.4f}')
    logger.info(f'Test time: {evaluate_time:.4f}s')
    logger.info(f'Average test time per sample: {evaluate_time / test_len:.4f}s')

    if inference_sample:
        logger.info('Inferencing trained model...')
        # Testing model with one example from test set
        data, label = next(iter(test_loader))

        idx = 0
        data, label = data[idx], label.numpy()[idx]

        output = model(data.unsqueeze(0).to(device))

        logger.info('Predicted Label : ', output_label(get_pred(output.cpu().detach().numpy())[0], data_path.name))
        logger.info('GT label: ', output_label(label, data_path.name))


if __name__ == '__main__':
    training()
