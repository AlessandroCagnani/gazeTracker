import torch
import torchvision


class Model(torchvision.models.ResNet):
    def __init__(self):
        block = torchvision.models.resnet.BasicBlock
        layers = [2, 2, 2] + [1]
        super().__init__(block, layers)
        del self.layer4
        del self.avgpool
        del self.fc

        pretrained_name = "resnet18"
        if pretrained_name:
            # Load the appropriate ResNet model using the available functions
            pretrained_model = getattr(torchvision.models, pretrained_name)(pretrained=True)
            # Get the state dict from the loaded pretrained model
            state_dict = pretrained_model.state_dict()

            self.load_state_dict(state_dict, strict=False)
            # While the pretrained models of torchvision are trained
            # using images with RGB channel order, in this repository
            # images are treated as BGR channel order.
            # Therefore, reverse the channel order of the first
            # convolutional layer.
            module = self.conv1
            module.weight.data = module.weight.data[:, [2, 1, 0]]

        with torch.no_grad():
            data = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
            features = self.forward(data)
            self.n_features = features.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
