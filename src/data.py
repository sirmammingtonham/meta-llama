from datasets import Dataset
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase
from torch.utils.data import DataLoader

import torch
import numpy as np
import itertools


def preprocess_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    method="direct",
    num_procs=8,
) -> Dataset:
    def preprocess_function(examples):
        """
        * tokenizes dataset
        * token_type_ids are 1 where there are label tokens and 0 otherwise
        """
        inputs = tokenizer(
            [inputs + " \n " for inputs in examples["inputs"]], add_special_tokens=False
        )
        targets = tokenizer(
            [" ".join(targets) + " \n\n " for targets in examples["targets"]],
            add_special_tokens=False,
        )

        # flip the location of inputs and targets for channel method
        if method == "channel":
            inputs, targets = targets, inputs

        outputs = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
        }

        for i in range(len(inputs["input_ids"])):
            outputs["input_ids"].append(
                inputs["input_ids"][i] + targets["input_ids"][i]
            )
            outputs["attention_mask"].append(
                inputs["attention_mask"][i] + targets["attention_mask"][i]
            )
            outputs["token_type_ids"].append(
                [0] * len(inputs["input_ids"][i])
                + [1] * len(targets["input_ids"][i])
                + [0]
            )

        return outputs

    ds = ds.map(
        preprocess_function,
        remove_columns=ds.column_names,
        batched=True,
        num_proc=num_procs,
    )
    return ds


@dataclass
class ICLCollator:
    tokenizer: PreTrainedTokenizerBase
    k_examples: int = 16
    max_length: int = 1024
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        * creates batches for in context/few shot learning
        * length of [features] should be (k_examples * batch_size)
        """
        batch = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

        for i in range(0, len(features), self.k_examples):
            batch["input_ids"].append(
                list(
                    itertools.chain.from_iterable(
                        example["input_ids"]
                        for example in features[i : i + self.k_examples]
                    )
                )
            )
            batch["attention_mask"].append(
                list(
                    itertools.chain.from_iterable(
                        example["attention_mask"]
                        for example in features[i : i + self.k_examples]
                    )
                )
            )
            batch["token_type_ids"].append(
                list(
                    itertools.chain.from_iterable(
                        example["token_type_ids"]
                        for example in features[i : i + self.k_examples]
                    )
                )
            )

        batch = self.tokenizer.pad(
            batch,
            padding="longest",
            max_length=self.max_length,
            pad_to_multiple_of=None,
            return_tensors=self.return_tensors,
        )

        return batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict: Dict[str, DataLoader], p=1):
        self.dataloader_dict = dataloader_dict
        N = max([len(x) ** (1 - p) for x in dataloader_dict.values()])
        f_p = lambda x: int(N * x**p)

        self.num_batches_dict = {
            task_name: f_p(len(dataloader))
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            f_p(len(dataloader.dataset)) for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])
