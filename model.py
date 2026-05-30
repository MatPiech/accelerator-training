import timm
import torch


def remove_sequential(network: torch.nn.Module, all_layers: list[torch.nn.Module]):
    for layer in network.children():
        # if sequential layer, apply recursively to layers in sequential layer
        if isinstance(layer, torch.nn.Sequential):
            # print(layer)
            remove_sequential(layer, all_layers)
        else:  # if leaf node, add it to list
            # print(layer)
            all_layers.append(layer)


def check_requires_grad(all_layers: list[torch.nn.Module], frozen_layer_num: int):
    print(f"Number of layers: {len(all_layers)}")
    print("Requires grad:")
    num_layer_requires_grad, num_trainable_blocks = 0, 0
    for idx, layer in enumerate(all_layers):
        layer_requires_grad = any([param.requires_grad for param in layer.parameters()])
        if idx <= frozen_layer_num:
            print(
                f'\t{(str(idx+1)+".").ljust(4)} {(layer.__class__.__name__+":").ljust(25)} {layer_requires_grad}\t->\tfrozen'
            )
        elif not layer_requires_grad:
            print(
                f'\t{(str(idx+1)+".").ljust(4)} {(layer.__class__.__name__+":").ljust(25)} {layer_requires_grad}\t->\tnot trainable'
            )
        else:
            print(
                f'\t{(str(idx+1)+".").ljust(4)} {(layer.__class__.__name__+":").ljust(25)} {layer_requires_grad}\t->\ttrainable'
            )
            num_trainable_blocks += 1
        num_layer_requires_grad += int(layer_requires_grad)
    print(f"Number of layers requiring grad: {num_layer_requires_grad}")
    print(f"Number of trainable blocks: {num_trainable_blocks}")


class Model(torch.nn.Module):
    def __init__(self, model_name: str, num_classes: int, channels: int, frozen_layer_num: int, img_size: int | None = None):
        super().__init__()
        if model_name == "visformer_tiny":
            model = timm.create_model(model_name, pretrained=True, in_chans=channels, img_size=img_size)
        else:
            model = timm.create_model(model_name, pretrained=True, in_chans=channels)

        all_layers: list[torch.nn.Module] = []
        remove_sequential(model, all_layers)
        check_requires_grad(all_layers, frozen_layer_num)
        features = all_layers[-1].in_features

        frozen_list: list[torch.nn.Module] = []
        end_list: list[torch.nn.Module] = []
        for i, layer in enumerate(all_layers[:-1]):
            if i <= frozen_layer_num:
                frozen_list.append(layer)
            else:
                end_list.append(layer)

        self.frozen_features = torch.nn.Sequential(*frozen_list)
        self.end_features = torch.nn.Sequential(*end_list)

        self.output = torch.nn.Linear(features, num_classes)

    def forward(self, x):
        x = self.frozen_features(x)
        x = self.end_features(x)
        logits = self.output(x)

        return logits
