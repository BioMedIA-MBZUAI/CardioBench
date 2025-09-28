from typing import Optional
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .dataset import EchoVideoCSV


class EchoCSVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        preprocess_val,
        *,
        root: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 8,
        prefetch_factor: int = 4,
        res=(224, 224),
        max_frames: int = 1,
        stride: int = 1,
        default_key_frame: int = 0,
        view: Optional[str] = None,
        modality: Optional[str] = None,
        key_frame_col: str = "key_frame",
        label_col: Optional[str] = None,
        drop_last: bool = True,
        random_single_frame: bool = False,
    ):
        super().__init__()
        self.csv_path, self.preprocess_val, self.root = csv_path, preprocess_val, root
        self.bs, self.nw = batch_size, num_workers
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last
        self.kw = dict(
            root=root,
            res=res,
            max_frames=max_frames,
            stride=stride,
            default_key_frame=default_key_frame,
            preprocess_val=preprocess_val,
            view=view,
            modality=modality,
            key_frame_col=key_frame_col,
            label_col=label_col,
            random_single_frame=random_single_frame,
        )
        self.ds_train = self.ds_val = self.ds_test = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.ds_train = EchoVideoCSV(self.csv_path, split="train", **self.kw)
            try:
                self.ds_val = EchoVideoCSV(self.csv_path, split="val", **self.kw)
            except FileNotFoundError:
                self.ds_val = None
        if stage in (None, "test", "validate", "predict"):
            try:
                self.ds_test = EchoVideoCSV(self.csv_path, split="test", **self.kw)
            except FileNotFoundError:
                self.ds_test = None

    def _dataloader(self, dataset, *, shuffle=False, drop_last=False):
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=self.bs,
            shuffle=shuffle,
            num_workers=self.nw,
            pin_memory=True,
            pin_memory_device="cuda",
            persistent_workers=(self.nw > 0),
            prefetch_factor=(self.prefetch_factor if self.nw > 0 else None),
            drop_last=drop_last,
        )

    def train_dataloader(self):
        return self._dataloader(self.ds_train, shuffle=True, drop_last=self.drop_last)

    def val_dataloader(self):
        return self._dataloader(self.ds_val)

    def test_dataloader(self):
        return self._dataloader(self.ds_test)

    def predict_dataloader(self):
        return self._dataloader(self.ds_test)
