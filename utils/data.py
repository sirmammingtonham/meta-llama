import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from typing import List


def init_clean_train_data(
    tasks: str, ignored_tasks: str, size=250, target_dir="clean_train_data"
) -> List[str]:
    """
    initialize training data directory with no generated examples
    """
    missing_tasks = []
    for task in tqdm(tasks):
        path = f"{target_dir}/{task}.json"

        if task in ignored_tasks:
            continue
        if os.path.exists(os.path.join(os.getcwd(), path)):  # already loaded
            continue
        try:
            D_task = load_dataset("tasksource/bigbench", task, split="train")
            k_idx = np.random.randint(
                low=0, high=len(D_task), size=min(size, len(D_task))
            ).tolist()
            D_init = D_task.select(k_idx)
            is_generated = [False] * len(D_init)
            D_init = D_init.add_column("is_generated", is_generated)
            D_init = D_init.add_column("true_idx", k_idx)
            D_init = D_init.remove_columns("idx")
            D_init.to_json(path, orient="records", lines=False)
        except:
            print(f"missing task: {task}")
            missing_tasks.append(task)

    return missing_tasks


def init_test_data(
    tasks: str, ignored_tasks: str, size=250, target_dir="clean_test_data"
) -> List[str]:
    """
    initialize test data directory
    """
    missing_tasks = []
    for task in tqdm(tasks):
        path = f"{target_dir}/{task}.json"

        if task in ignored_tasks:
            continue
        if os.path.exists(os.path.join(os.getcwd(), path)):  # already loaded
            continue
        try:
            D_task = load_dataset("tasksource/bigbench", task, split="train")
            k_idx = np.random.randint(
                low=0, high=len(D_task), size=min(size, len(D_task))
            ).tolist()
            D_init = D_task.select(k_idx)
            is_generated = [False] * len(D_init)
            D_init = D_init.add_column("is_generated", is_generated)
            D_init = D_init.add_column("true_idx", k_idx)
            D_init = D_init.remove_columns("idx")
            D_init.to_json(path, orient="records", lines=False)
        except:
            print(f"missing task test: {task}")
            missing_tasks.append(task)

    return missing_tasks
