from base_model import PlBaseModel
import torch.nn as nn
from utils_model import ResidualBlock
import torch.nn.functional as F

class TextDetectLSTM(nn.Module):

    def __init__(self, input_dim, output_dim, activation, dropout):
        super(self).__init__()

        self.conv1 = nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.residual1 = ResidualBlock(16, 16, activation, True, 1, dropout)
        self.residual2 = ResidualBlock(16, 16, activation, True, 2, dropout)
        self.residual3 = ResidualBlock(16, 16, activation, False, 1, dropout)

        self.residual4 = ResidualBlock(16, 32, activation, True, 2, dropout)
        self.residual5 = ResidualBlock(32, 32, activation, False, 1, dropout)

        self.residual6 = ResidualBlock(32, 64, activation, True, 1, dropout)
        self.residual7 = ResidualBlock(64, 64, activation, False, 1, dropout)

        self.blstm = nn.LSTM(64, 64, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(2 * 64, output_dim + 1)

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

        x = x.view(x.size(0), -1)
        x, _ = self.blstm(x)
        x = self.dropout(x)
        x = self.fc(x)

        return F.softmax(x, dim=2)

    def get_loss_fn(self):
        pass
