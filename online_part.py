from pathlib import Path

from onnxruntime.training.api import CheckpointState, Module, Optimizer
from onnxruntime import InferenceSession
import numpy as np
import torch
import torchvision
import evaluate
from tqdm import tqdm


def get_data_loaders():
    batch_size = 64

    norm_mean = 0.5
    norm_std = 0.5

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((norm_mean, ), (norm_std, ))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


# Util function to convert logits to predictions.
def get_pred(logits):
    return np.argmax(logits, axis=1)


def output_label(label):
    output_mapping = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]


def online_training():
    train_loader, test_loader = get_data_loaders()

    # artifacts path
    artifacts_path = Path('./artifacts')

    # Create checkpoint state.
    state = CheckpointState.load_checkpoint(artifacts_path / 'checkpoint')

    # Create module.
    model = Module(artifacts_path / 'training_model.onnx', state, artifacts_path / 'eval_model.onnx', device='cpu')

    # Create optimizer.
    optimizer = Optimizer(artifacts_path / 'optimizer_model.onnx', model)

    epochs = 5
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1} / {epochs}')
        # Training Loop
        model.train()
        losses = []
        for _, (data, target) in enumerate(tqdm(train_loader)):
            forward_inputs = [data.numpy(), target.numpy().astype(np.int64)]
            train_loss, _ = model(*forward_inputs)
            optimizer.step()
            model.lazy_reset_grad()
            losses.append(train_loss)

        print(f'Epoch: {epoch+1}, Train Loss: {sum(losses)/len(losses):.4f}')
        
        # Test Loop
        model.eval()
        losses = []
        metric = evaluate.load('accuracy')

        for _, (data, target) in enumerate(tqdm(test_loader)):
            forward_inputs = [data.numpy(), target.numpy().astype(np.int64)]
            test_loss, logits = model(*forward_inputs)
            metric.add_batch(references=target, predictions=get_pred(logits))
            losses.append(test_loss)

        metrics = metric.compute()
        print(f'Epoch: {epoch+1}, Test Loss: {sum(losses)/len(losses):.4f}, Accuracy : {metrics["accuracy"]:.2f}')

    model.export_model_for_inferencing(artifacts_path / 'inference_model.onnx', ['output'])
    session = InferenceSession(artifacts_path / 'inference_model.onnx', providers=['CPUExecutionProvider'])

    # getting one example from test list to try inference.
    data, label = next(iter(test_loader))

    idx = 0
    data, label = data[idx], label.numpy()[idx]

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name 
    output = session.run([output_name], {input_name: data.numpy()[np.newaxis]})

    print('Predicted Label : ', output_label(get_pred(output[0])[0]))
    print('GT label: ', output_label(label))


if __name__ == '__main__':
    online_training()
