import os.path

from torch.utils.data import Dataset
import tqdm
import random
from Data.utils_data import ImagePreProcess,LabelTokenizer
import cv2


class OCRDataset(Dataset):

    def __init__(self, annotation_path: str, **kwargs) -> None:
        self.max_len = None
        self.vocab = None
        self.dataset = None
        self.annotation_path = annotation_path
        self.base_path = "/".join(annotation_path.replace("\\", "/").split("/")[:-3])
        self.shuffle = kwargs.pop("ds_shuffle", False)
        self.image_size = kwargs.pop("ds_image_size", [100, 200])
        self.transformers = kwargs.pop("ds_transformers", None)
        self.read_annotation_data()
        if self.transformers is None:
            self.transformers=self.get_default_transformers()
            for transformer in self.transformers:
                if isinstance(transformer, LabelTokenizer):
                    self.vocab=transformer.get_vocab()

        if self.shuffle:
            random.shuffle(self.dataset)



    def read_annotation_data(self):
        labels,dataset, vocab, max_len = [],[], set(), 0

        with open(self.annotation_path, "r") as f:
            for line in tqdm.tqdm(f.readlines(), desc="Processing lines"):
                line = line.split(";")
                image_path = line[0]
                label = line[1].strip().strip("\n")
                dataset.append([self.base_path + image_path, label])
                labels.append(label)
                vocab.update(list(label))
                max_len = max(max_len, len(label))
        self.dataset = dataset
        self.vocab = vocab
        self.max_len = max_len
        self.labels=labels

    def get_default_transformers(self):
        _ = [
            ImagePreProcess(width=self.image_size[0], height=self.image_size[1]),
            LabelTokenizer(vocab=self.vocab,max_length=self.max_len)
        ]
        return _

    def __getitem__(self, index: int):
        path, label = self.dataset[index]
        img = cv2.imread(path)
        if self.transformers is not None:
            for tr in self.transformers:
                img, label = tr(img, label)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def show_example(self, number_of_examples: int = 1):
        for _ in range(0, number_of_examples):
            img_path, label = random.choice(self.dataset)
            img = cv2.imread(img_path)
            cv2.imshow(f"Label: {label}", img)
            cv2.waitKey(0)


if __name__ == "__main__":
    data = OCRDataset(annotation_path=r"D:\data_ocr\labels\val\labels.txt")
    data.show_example(number_of_examples=10)
