import torch.nn as nn
from Models.architectures.utils_model import ResidualBlock, activation_layer
import torch.nn.functional as F


class TextDetectLSTM(nn.Module):

    def __init__(self, input_dim, output_dim, activation, dropout, **kwargs):
        super().__init__()
        leaky_alpha = kwargs.pop("leaky_alpha", 0.1)
        self.conv1 = nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = activation_layer(activation=activation, alpha=leaky_alpha, inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.residual1 = ResidualBlock(16, 16, activation=activation, leaky_alpha=leaky_alpha, dropout=dropout,stride=2)
        self.residual2 = ResidualBlock(16, 16, activation=activation, leaky_alpha=leaky_alpha, dropout=dropout,stride=2)
        self.residual3 = ResidualBlock(16, 16, activation=activation, leaky_alpha=leaky_alpha, dropout=dropout)

        self.residual4 = ResidualBlock(16, 32, activation=activation, leaky_alpha=leaky_alpha, dropout=dropout)
        self.residual5 = ResidualBlock(32, 32, activation=activation, leaky_alpha=leaky_alpha, dropout=dropout)

        self.residual6 = ResidualBlock(32, 64, activation=activation, leaky_alpha=leaky_alpha, dropout=dropout)
        self.residual7 = ResidualBlock(64, 64, activation=activation, leaky_alpha=leaky_alpha, dropout=dropout)

        self.blstm = nn.LSTM(64, 128, bidirectional=True, batch_first=True,num_layers=1)

        self.fc = nn.Linear(2 * 128, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)

        x = self.residual4(x)
        x = self.residual5(x)

        x = self.residual6(x)
        x = self.residual7(x)

        x =  x.reshape(x.size(0), -1, x.size(1))
        x, _ = self.blstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        x=F.log_softmax(x, 2)

        return x
