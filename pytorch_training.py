from pathlib import Path

import click
import numpy as np
import torch
import torchvision
from tqdm import tqdm


def get_data_loaders(data_path: Path, batch_size: int, norm_mean: tuple[float] = (0.5,), norm_std: tuple[float]  = (0.5,)):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(norm_mean, norm_std)
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = corrects / batch_size
    return accuracy.item()


# Util function to convert logits to predictions.
def get_pred(logits):
    return np.argmax(logits, axis=1)


def output_label(label):
    cifar10_mapping = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    input = (label.item() if type(label) == torch.Tensor else label)
    return cifar10_mapping[input]


@click.command()
@click.option('--model-path', type=click.Path(exists=True, file_okay=True), required=True)
@click.option('--data-path', type=click.Path(exists=True), required=True)
@click.option('--device', type=click.Choice(['cpu', 'cuda']), default='cpu')
@click.option('--batch-size', type=int, default=64)
def training(model_path: Path, data_path: Path, device: str, batch_size: int):
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

    print('Setting criterion...')
    criterion = torch.nn.CrossEntropyLoss()

    print('Creating optimizer..')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    epochs = 5
    print(f'Starting training loop for {epochs} epochs...')
    for epoch in range(epochs):
        model = model.train()

        train_running_loss = 0.0
        train_acc = 0.0

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
            train_acc += get_accuracy(logits, labels, 32)

        model.eval()

        test_running_loss = 0.0
        test_acc = 0.0

        with torch.no_grad():
            for k, (images, labels) in enumerate(tqdm(test_loader)):
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                test_running_loss += loss.detach().item()
                test_acc += get_accuracy(logits, labels, 32)

        print(f'Epoch: {epoch} |',
            f' Loss: {train_running_loss/i:.4f} | Train Accuracy: {train_acc/i:.2f} |',
            f' Test loss: {test_running_loss/k:.4f} | Test Accuracy: {test_acc/k:.2f}')
        
    # Testing model with one example from test set
    data, label = next(iter(test_loader))

    idx = 0
    data, label = data[idx], label.numpy()[idx]

    output = model(data.unsqueeze(0).to(device))

    print('Predicted Label : ', output_label(get_pred(output.cpu().detach().numpy())[0]))
    print('GT label: ', output_label(label))


if __name__ == '__main__':
    training()
