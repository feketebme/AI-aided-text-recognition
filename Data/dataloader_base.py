import copy
import typing
from torch.utils.data import Dataset, random_split, DataLoader
import pytorch_lightning as pl
from Data.utils_data import ImagePreProcess

class DataProvider(pl.LightningDataModule):
    available_datasets = []

    def __init__(self, annotation_paths: typing.List[str], dataset_object: typing.Type[Dataset], **kwargs):
        """
                Base pytorch lightning dataloader

                :param annotation_paths: List of paths to annotation files.
                :param dataset_object: Type of dataset to be used.
                :param split_dataset: List containing proportions for train, val, and test splits (default: [0.8, 0.1, 0.1]).
                :param shuffle: Flag for shuffling the dataset (default: False).
                :param batch_size: Batch size for DataLoader (default: 1).
                :param image_size: Default image size for processing (default: [100, 200]).
                :param transformers: List of image transformers (default: []).
        """
        super(DataProvider, self).__init__()
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.annotation_paths = annotation_paths
        self.dataset_object = dataset_object
        self.split_dataset = kwargs.pop("split_dataset", [0.8, 0.1, 0.1])
        self.batch_size = kwargs.pop("batch_size", 1)
        ds_shuffle = kwargs.pop("shuffle", False)
        ds_image_size = kwargs.pop("image_size", [100, 200])
        ds_transformers = kwargs.pop("transformers", None)
        self.ds_params = {}
        ds_params_update = {k: v for k, v in  locals().copy().items() if 'ds' in k}
        self.ds_params.update(ds_params_update)
        self.prepare()

    @staticmethod
    def get_default_transformers(image_size):
        _ = [ImagePreProcess(height=image_size[0], width=image_size[1])]
        return _
    def prepare(self):
        for path in self.annotation_paths:
            self.available_datasets.append(self.dataset_object(path, **self.ds_params))
        if len(self.available_datasets) < 2:
            self.available_datasets = random_split(self.available_datasets[0], self.split_dataset)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self.available_datasets[0]
            self.val_dataset = self.available_datasets[1]
        if stage == "test" and len(self.available_datasets) == 3:
            self.test_dataset = self.available_datasets[2]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,num_workers=9)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
