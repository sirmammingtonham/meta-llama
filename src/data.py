from datasets import Dataset, load_dataset
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase
from torch.utils import data

import numpy as np
import itertools


def load_datasets(
    dataset_names: List[str],
    data_dir: str,
    use_augmented=False,
    preprocess_fn=lambda x: x,
) -> Dict[str, Dataset]:
    if use_augmented:
        return {
            name: preprocess_fn(
                load_dataset(
                    "json",
                    data_files=f"{data_dir}/{name}.json",
                )
            )
            for name in dataset_names
        }
    else:
        return {
            name: preprocess_fn(load_dataset(f"tasksource/bigbench", name))
            for name in dataset_names
        }


def preprocess_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    method="direct",
    num_procs=8,
) -> Dataset:
    def remove_choices(s: str) -> str:
        choice_start = s.find("\n  choice:")
        if choice_start == -1:
            return s
        return s[:choice_start]

    def target_to_index(choices: List, target: List[str]) -> int:
        if not target:
            return -1
        try:
            index = choices.index(target[0])
            return index
        except ValueError:
            return -1

    def preprocess_function(examples: dict) -> dict:
        """
        * tokenizes dataset
        * token_type_ids are 1 where there are label tokens and 0 otherwise
        """
        inputs = [
            f"{remove_choices(inp)}\n"
            + "\n".join([f"choice {i}: {choice}" for i, choice in enumerate(choices)])
            + "\nanswer:"
            for inp, choices in zip(
                examples["inputs"], examples["multiple_choice_targets"]
            )
        ]

        targets = [
            f"{target_to_index(choices, target)}\n\n"
            for target, choices in zip(
                examples["targets"], examples["multiple_choice_targets"]
            )
        ]

        # swap inputs and targets if method is "channel"
        if method == "channel":
            inputs, targets = targets, inputs

        # tokenize inputs and targets, and prepare outputs dictionary
        input_tokenized = tokenizer(inputs, add_special_tokens=False)
        target_tokenized = tokenizer(targets, add_special_tokens=False)
        outputs = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
        }

        # merge input and target tokens and prepare outputs
        for i in range(len(input_tokenized["input_ids"])):
            input_ids = input_tokenized["input_ids"][i]
            target_ids = target_tokenized["input_ids"][i]
            outputs["input_ids"].append(input_ids + target_ids)

            input_attention = input_tokenized["attention_mask"][i]
            target_attention = target_tokenized["attention_mask"][i]
            outputs["attention_mask"].append(input_attention + target_attention)

            input_token_type = [0] * (len(input_ids) + 1)
            target_token_type = [1] * (len(target_ids) - 3) + [0, 0]
            outputs["token_type_ids"].append(input_token_type + target_token_type)

        return outputs

    ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=num_procs,
    )
    return ds


@dataclass
class ICLCollator:
    tokenizer: PreTrainedTokenizerBase
    k_examples: int = 16
    max_length: int = 2048
    return_tensors: str = "pt"
    for_eval: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        * creates batches for in context/few shot learning
        * length of [features] should be (k_examples * batch_size)
        * if for_eval create a labels field
        """
        batch = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

        if self.for_eval:
            # if collation for evaluation, features is a List[List[Dict[str, Any]]]
            # where the inner list contains our k_examples, so flatten it
            features = list(itertools.chain.from_iterable(features))

        for i in range(0, len(features), self.k_examples):
            batch["input_ids"].append(
                list(
                    itertools.chain.from_iterable(
                        example["input_ids"]
                        for example in features[i : i + self.k_examples]
                    )
                )[: self.max_length]
            )
            batch["attention_mask"].append(
                list(
                    itertools.chain.from_iterable(
                        example["attention_mask"]
                        for example in features[i : i + self.k_examples]
                    )
                )[: self.max_length]
            )
            batch["token_type_ids"].append(
                list(
                    itertools.chain.from_iterable(
                        example["token_type_ids"]
                        for example in features[i : i + self.k_examples]
                    )
                )[: self.max_length]
            )

        batch = self.tokenizer.pad(
            batch,
            padding="longest",
            max_length=self.max_length,
            pad_to_multiple_of=None,
            return_tensors=self.return_tensors,
        )

        if self.for_eval:
            batch["labels"] = batch["input_ids"].clone()
            batch["labels"] *= batch["token_type_ids"]

        return batch


@dataclass
class EvalDatasetWrapper(data.Dataset):
    """
    Simple Dataset wrapper that returns k_examples-1 random
    examples from the training set for each evaluation example
    """

    train_dataset: Dataset
    eval_dataset: Dataset
    k_examples: int = 16

    def __len__(self):
        return len(self.eval_dataset)

    def __getitem__(self, index):
        random_examples = np.random.randint(
            0, len(self.train_dataset), size=(self.k_examples - 1,)
        )
        examples = [self.train_dataset[i.item()] for i in random_examples]
        for x in examples:
            # ignore label mask for the examples, we only care about the last one
            x["token_type_ids"] = [0] * len(x["token_type_ids"])

        target = self.eval_dataset[index]

        return examples + [target]


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict: Dict[str, data.DataLoader], p=1):
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
