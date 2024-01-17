import torch
import torch.nn as nn
import cv2
import typing
import numpy as np
from torchvision.transforms import ToTensor
from torchtext.vocab import build_vocab_from_iterator
from torch.nn import ConstantPad1d
from itertools import groupby
from torchmetrics.text import CharErrorRate
from collections import OrderedDict

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x},{self.y})"


def do_overlap(l1, r1, l2, r2):
    # If one rectangle is on left side of other
    if l1.x >= r2.x or l2.x >= r1.x:
        return False

    # If one rectangle is above other
    if l1.y <= r2.y or l2.y <= r1.y:
        return False

    return True


class LabelTokenizer:
    """
        This class provides functionality to tokenize and pad label data.
        Args:
            vocab (set): Vocabulary set containing characters.
            max_length (int): Maximum length of labels.
    """

    def __init__(self, vocab, max_length):
        # Build ordered vocabulary
        self.vocab = {char: idx+1 for idx, char in enumerate(sorted(vocab))} # Shift indices by 1
        self.vocab['<pad>'] = 0  # Set <pad> value to 0
        self.max_length = max_length

    def __call__(self, image, label):
        # Tokenize and encode data
        encoded_data = [self.vocab.get(token, self.vocab['<pad>']) for token in label]
        # Pad the encoded data
        padded_data = ConstantPad1d((0, self.max_length - len(label)), self.vocab['<pad>'])(torch.tensor(encoded_data))
        return image, padded_data

    def decode(self, tensor):
        """
        Decode a tensor into its original string representation.
        Args:
            tensor (Tensor): Padded tensor to be decoded.
        Returns:
            str: The decoded string.
        """
        decoded_tokens = [char for idx in tensor for char, idx_vocab in self.vocab.items() if idx == idx_vocab and char != '<pad>']
        return ''.join(decoded_tokens)
    def get_vocab(self):
        """
        Get the vocabulary set.
        Returns:
            dict: The vocabulary set.
        """
        return self.vocab


class ImagePreProcess:
    """
        ImageResizer class for resizing images using OpenCV. Resize and normalize the input image.

        Attributes:
            width (int): Target width for resizing.
            height (int): Target height for resizing.
        """

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.to_tensor = ToTensor()

    def __call__(self, image: np.ndarray, label: typing.Any) -> typing.Tuple[torch.Tensor, typing.Any]:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected image to be of type numpy.ndarray, got {type(image)}")

        if isinstance(image, np.ndarray):
            image_numpy = cv2.resize(image, (self.width, self.height))
            image_normalized = image_numpy / 255.0  # Normalize pixel values
            image_tensor = torch.from_numpy(image_normalized).permute(2, 1, 0).float()
            return image_tensor, label


class CTCLoss(nn.Module):
    """ CTC loss for PyTorch
    """

    def __init__(self, blank: int, reduction: str = "mean", zero_infinity: bool = False):
        """ CTC loss for PyTorch

        Args:
            blank: Index of the blank label
        """
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
        self.blank = blank

    def forward(self, output, target):
        """
        Args:
            output: Tensor of shape (batch_size, sequence_length, num_classes)
            target: Tensor of shape (batch_size, sequence_length)

        Returns:
            loss: Scalar
        """
        # Remove padding and blank tokens from target
        target_lengths = torch.sum(target != self.blank, dim=1)
        using_dtype = torch.long
        device = output.device

        target_unpadded = target[target != self.blank].view(-1).to(using_dtype)

        output = output.permute(1, 0, 2)  # (sequence_length, batch_size, num_classes)
        output_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=using_dtype).to(device)

        loss = self.ctc_loss(output, target_unpadded, output_lengths, target_lengths.to(using_dtype))

        return loss


class CERMetrics:

    def __init__(
            self,
            vocabulary: typing.Union[str, list],
    ) -> None:
        self.vocab = vocabulary
        self.metr=CharErrorRate()

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        """
           Update metric state with new data.
           Args:
             output (torch.Tensor): Output of model.
             target (torch.Tensor): Target data.
        """
        # Convert to numpy
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        # Get predicted class indices
        pred_indices = np.argmax(output, axis=-1)
        target_indices = target

        # Use groupby to find continuous same indices
        grouped_preds = [[k for k, _ in groupby(preds)] for preds in pred_indices]

        itos = {v: k for k, v in self.vocab.items()}

        # Convert indices to strings
        output_texts = ["".join([itos.get(k, '') for k in group if k < len(itos)]) for group in grouped_preds]
        target_texts = ["".join([itos.get(k, '') for k in group if k < len(itos)]) for group in target_indices]
        return self.metr(output_texts, target_texts)
