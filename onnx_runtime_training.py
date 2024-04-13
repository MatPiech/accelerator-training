from pathlib import Path

from onnxruntime.training.api import CheckpointState, Module, Optimizer
from onnxruntime import InferenceSession
import numpy as np
import torch
import torchvision
from tqdm import tqdm


def get_data_loaders(batch_size: int, norm_mean: tuple[float] = (0.5,), norm_std: tuple[float]  = (0.5,)):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(norm_mean, norm_std)
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (get_pred(logit) == target.numpy()).sum()
    accuracy = corrects / batch_size
    return accuracy.item()


# Util function to convert logits to predictions.
def get_pred(logits):
    return np.argmax(logits, axis=1)


def output_label(label):
    cifar10_mapping = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    input = (label.item() if type(label) == torch.Tensor else label)
    return cifar10_mapping[input]


def training():
    print('Creating dataloaders...')
    batch_size = 64
    train_loader, test_loader = get_data_loaders(batch_size)

    # artifacts path
    artifacts_path = Path('./artifacts')

    print('Loading artifacts...')
    # Create checkpoint state
    state = CheckpointState.load_checkpoint(artifacts_path / 'checkpoint')

    # Create module
    device = 'cuda'
    model = Module(artifacts_path / 'training_model.onnx', state, artifacts_path / 'eval_model.onnx', device=device)

    # Create optimizer
    optimizer = Optimizer(artifacts_path / 'optimizer_model.onnx', model)

    epochs = 5
    print(f'Starting training loop for {epochs} epochs...')
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1} / {epochs}')
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

        print(f'Epoch: {epoch} |',
          f' Loss: {sum(train_losses) / len(train_losses):.4f} | Train Accuracy: {sum(train_acc) / len(train_acc):.2f} |',
          f' Test loss: {sum(test_losses) / len(test_losses):.4f} | Test Accuracy: {sum(test_acc) / len(test_acc):.2f}')

    model.export_model_for_inferencing(artifacts_path / 'inference_model.onnx', ['output'])
    session = InferenceSession(artifacts_path / 'inference_model.onnx', providers=['CPUExecutionProvider'])

    # Testing model with one example from test set
    data, label = next(iter(test_loader))

    idx = 0
    data, label = data[idx], label.numpy()[idx]

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name 
    output = session.run([output_name], {input_name: data.numpy()[np.newaxis]})

    print('Predicted Label : ', output_label(get_pred(output[0])[0]))
    print('GT label: ', output_label(label))


if __name__ == '__main__':
    training()
