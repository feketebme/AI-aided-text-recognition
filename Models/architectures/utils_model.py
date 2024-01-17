import torch.nn as nn
import torch


def activation_layer(activation: str = "relu", alpha: float = 0.1, inplace: bool = True):
    """ Activation layer wrapper for various activation functions

    Args:
        activation: str, activation function name (default: 'relu')
        alpha: float (LeakyReLU activation function parameter)

    Returns:
        torch.Tensor: activation layer
    """
    if activation.lower() == "relu":
        return nn.ReLU(inplace=inplace)

    elif activation.lower() == "leaky_relu":
        return nn.LeakyReLU(negative_slope=alpha, inplace=inplace)

    elif activation.lower() == "sigmoid":
        return nn.Sigmoid()

    elif activation.lower() == "tanh":
        return nn.Tanh()

    elif activation.lower() == "softmax":
        return nn.Softmax(dim=-1)

    elif activation.lower() == "elu":
        return nn.ELU(alpha=alpha, inplace=inplace)

    elif activation.lower() == "selu":
        return nn.SELU(inplace=inplace)

    # Add more activation functions as needed

    else:
        raise ValueError(f"Unsupported activation function: {activation}")


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out1x1, reduce3x3, out3x3, reduce5x5, out5x5, out1x1pool, activation='Relu',leaky_alpha=0.1,inplace=True):
        super(InceptionBlock, self).__init__()

        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out1x1, kernel_size=1)

        # 1x1 convolution followed by 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce3x3, kernel_size=1),
            activation_layer(activation='Relu',leaky_alpha=0.1,inplace=True),
            nn.Conv2d(reduce3x3, out3x3, kernel_size=3, padding=1)
        )

        # 1x1 convolution followed by 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce5x5, kernel_size=1),
            activation_layer(activation='Relu', leaky_alpha=0.1, inplace=True),
            nn.Conv2d(reduce5x5, out5x5, kernel_size=5, padding=2)
        )

        # 3x3 max pooling followed by 1x1 convolution branch
        self.branch1x1pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out1x1pool, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch1x1pool = self.branch1x1pool(x)

        # Concatenate along the channel dimension
        output = torch.cat([branch1x1, branch3x3, branch5x5, branch1x1pool], dim=1)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='leaky_relu',leaky_alpha=0.1, dropout=0.2,stride=1):
        super(ResidualBlock, self).__init__()

        self.activation = activation
        self.dropout = dropout

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = activation_layer(activation=activation,alpha=leaky_alpha,inplace=True)
        self.dropout = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.bn_shortcut = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = self.shortcut(residual)
        shortcut = self.bn_shortcut(shortcut)

        out += shortcut
        return out

