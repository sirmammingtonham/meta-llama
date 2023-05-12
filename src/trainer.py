from typing import Optional, List, Dict

from torch.utils.data import DataLoader
from transformers import Trainer, PreTrainedTokenizerBase
from datasets import Dataset

from .data import MultitaskDataloader, EvalDatasetWrapper, ICLCollator


class ICLTrainer(Trainer):
    """Wrap trainer to use our custom multitask dataloader."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        k_examples: int = 16,
        *args,
        **kwargs,
    ):
        super().__init__(tokenizer, *args, **kwargs)
        self.k_examples = k_examples

    def get_train_dataloader(self) -> DataLoader:
        train_datasets = self.train_dataset
        collate_fn = ICLCollator(self.tokenizer, k_examples=self.k_examples)

        loader_batch_size = self.k_examples * self.args.per_device_train_batch_size

        train_dataloader_dict = {
            k: DataLoader(
                v,
                batch_size=loader_batch_size,
                collate_fn=collate_fn,
                shuffle=True,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
            for k, v in train_datasets.items()
        }
        return MultitaskDataloader(train_dataloader_dict)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        dataset = EvalDatasetWrapper(
            eval_dataset["train"],
            eval_dataset["validation"],
            k_examples=self.k_examples,
        )

        collate_fn = ICLCollator(
            self.tokenizer,
            k_examples=self.k_examples,
            for_eval=True,
        )

        return DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_test_dataloader(self, test_dataset: Optional[Dataset] = None) -> DataLoader:
        dataset = EvalDatasetWrapper(
            test_dataset["train"],
            test_dataset["test"],
            k_examples=self.k_examples,
        )

        collate_fn = ICLCollator(
            self.tokenizer,
            k_examples=self.k_examples,
            for_eval=True,
        )

        return DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate(
        self,
        eval_datasets: Dict[str, Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluations on each dataset in eval datasets and returns metrics.
        """

        if eval_datasets is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_datasets = (
            eval_datasets if eval_datasets is not None else self.eval_dataset
        )

        metrics = {}

        for name, dataset in eval_datasets.items():
            metrics.update(
                super().evaluate(
                    dataset,
                    ignore_keys,
                    f"{metric_key_prefix}_{name}",
                )
            )
