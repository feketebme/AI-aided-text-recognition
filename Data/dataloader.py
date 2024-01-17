from abc import ABC

import cv2
import torch
from torch.utils.data import Dataset
import tqdm
import random
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.torch.dataProvider import DataProvider


class DataProvider(Dataset):

    def __init__(self, annotation_path: str, **kwargs):
        self.data_source = annotation_path
        self.data_path = ''.join(self.data_source.replace('\\', '/').split('/')[:-2])
        self.shuffle = kwargs.pop("shuffle", False)
        self.batch_size = kwargs.pop("batch_size", 1)
        self.transformers = kwargs.pop("transformers", None)
        self.image_size = kwargs.pop("image_size", [100, 200])
        self.dataset, self.vocab, self.max_len = self.read_annotation_file()
        if self.shuffle:
            random.shuffle(self.dataset)

    def read_annotation_file(self):
        dataset, vocab, max_len = [], set(), 0
        with open(self.annotation_path, "r") as f:
            for line in tqdm(f.readlines()):
                line = line.split()
                image_path = self.data_path + line[0][1:]
                label = line[0].split("_")[1]
                dataset.append([image_path, label])
                vocab.update(list(label))
                max_len = max(max_len, len(label))
        return dataset, sorted(vocab), max_len

    def __getitem__(self, index:int):
        for path, label in self.dataset:
            img = cv2.imread(path)

            if self.transformers is not None:
                pass
            yield img, label
