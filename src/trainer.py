from typing import Dict, Callable, Optional

from torch.utils.data import DataLoader
from transformers import Trainer
from datasets import Dataset

from .data import MultitaskDataloader


class ICLTrainer(Trainer):
    """Wrap trainer to use our custom multitask dataloader. (If not provided, default to regular trainer)"""

    def __init__(self, k_examples: int = 16, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.k_examples = k_examples

    def get_train_dataloader(self) -> DataLoader:
        train_datasets = self.train_dataset
        collate_fn = self.data_collator

        loader_batch_size = self.k_examples * self.args.per_device_train_batch_size

        train_dataloader_dict = {
            k: DataLoader(v, batch_size=loader_batch_size, collate_fn=collate_fn)
            for k, v in train_datasets.items()
        }
        return MultitaskDataloader(train_dataloader_dict)

    def get_eval_dataloader(
        self, eval_datasets: Optional[Dataset] = None
    ) -> DataLoader:
        eval_datasets = (
            eval_datasets if eval_datasets is not None else self.eval_dataset
        )
        collate_fn = self.data_collator

        loader_batch_size = self.k_examples * self.args.per_device_eval_batch_size

        eval_dataloader_dict = {
            k: DataLoader(v, batch_size=loader_batch_size, collate_fn=collate_fn)
            for k, v in eval_datasets.items()
        }
        return MultitaskDataloader(eval_dataloader_dict)

    def get_test_dataloader(
        self, test_datasets: Optional[Dataset] = None
    ) -> DataLoader:
        collate_fn = self.data_collator

        loader_batch_size = self.k_examples * self.args.per_device_eval_batch_size

        test_dataloader_dict = {
            k: DataLoader(v, batch_size=loader_batch_size, collate_fn=collate_fn)
            for k, v in test_datasets.items()
        }
        return MultitaskDataloader(test_dataloader_dict)
