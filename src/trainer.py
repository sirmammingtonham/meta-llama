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
        super().__init__(tokenizer=tokenizer, *args, **kwargs)
        self.k_examples = k_examples

    def get_train_dataloader(self) -> DataLoader:
        train_datasets = self.train_dataset
        collate_fn = ICLCollator(self.tokenizer, k_examples=self.k_examples)

        loader_batch_size = self.k_examples * self.args.per_device_train_batch_size

        train_dataloader_dict = {
            k: DataLoader(
                v["train"],
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
        eval_dataset = eval_dataset["train"].train_test_split(
            test_size=0.2
        )  # dont use the actual test set for validation
        dataset = EvalDatasetWrapper(
            eval_dataset["train"],
            eval_dataset["test"],
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
            test_dataset["validation"],
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
        eval_dataset: Dict[str, Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluations on each dataset in eval datasets and returns metrics.
        """

        metrics = super().evaluate(
            eval_dataset,
            ignore_keys,
            f"{metric_key_prefix}",
        )

        metrics.pop(f"{metric_key_prefix}_runtime")
        metrics.pop(f"{metric_key_prefix}_samples_per_second")
        metrics.pop(f"{metric_key_prefix}_steps_per_second")

        return metrics

    def predict(
        self,
        test_dataset: Dict[str, Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
    ) -> Dict[str, float]:
        """
        Run evaluations on each dataset in eval datasets and returns metrics.
        """

        if test_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        test_dataset = test_dataset if test_dataset is not None else self.eval_dataset

        metrics = {}

        for name, dataset in test_dataset.items():
            print("-" * 100)
            print(name)
            print("-" * 100)
            metrics.update(
                super()
                .predict(
                    dataset,
                    ignore_keys,
                    f"{metric_key_prefix}/{name}",
                )
                .metrics
            )
            metrics.pop(f"{metric_key_prefix}/{name}_loss")
            metrics.pop(f"{metric_key_prefix}/{name}_runtime")
            metrics.pop(f"{metric_key_prefix}/{name}_samples_per_second")
            metrics.pop(f"{metric_key_prefix}/{name}_steps_per_second")

        return metrics
