import csv
import os
import openai
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

np.random.seed(0)

EXCLUDED_TASKS = ["cifar10_classification", "mnist_ascii"]


class GenerationPipeline:
    def __init__(
        self, tasks_dir: str, train_dir: str, task_seed_size: int, sk: str
    ) -> None:
        """
        tasks_dir: name of file (csv) of tasks to augment
        train_dir: file path to directory to write data to
        k: amount of in-context examples for generation
        task_seed_size: size of seed examples per task
        sk: OpenAI api key
        """
        openai.api_key = sk

        with open(tasks_dir, newline="") as f:
            reader = csv.reader(f)
            tasks = [row[-1] for row in reader]
        tasks = tasks[1:]
        f.close()
        tasks = list(filter(lambda task: task not in EXCLUDED_TASKS, tasks))

        # train test split wrt tasks
        idx = np.random.permutation(len(tasks))
        split = int(len(tasks) * 0.8)  # 80, 20 split
        train_idx, test_idx = idx[:split], idx[split:]
        self.train_tasks = [tasks[i] for i in train_idx]
        self.test_tasks = [tasks[i] for i in test_idx]

        # initialize seed task jsons in training directory
        self.missing_tasks = self._init_task_seeds(task_seed_size, train_dir)

    def _init_task_seeds(self, task_seed_size: int, train_dir: str):
        """
        only augment training tasks
        train_dir: file path to directory to write data to
        task_seed_size: size of seed examples per task
        return missing tasks lists: [str]
        """
        missing_tasks = []
        for task in tqdm(self.train_tasks):
            path = f"{train_dir}/{task}.json"
            if os.path.exists(os.path.join(os.getcwd(), path)):  # already loaded
                continue
            try:
                D_task = load_dataset("tasksource/bigbench", task, split="train")
                k_idx = np.random.randint(
                    low=0, high=len(D_task), size=task_seed_size
                ).tolist()
                D_init = D_task.select(k_idx)
                D_init.to_json(path)  # write to json
            except:
                missing_tasks.append(task)
        return missing_tasks

    def augment(self, k: int, train_dir: str):
        """
        k: amount of in-context examples for generation
        train_dir: file path to directory containing train data json files

        for all train tasks:
        (0) Sample k examples from task
          * format k examples into prompt
        (1) GPT to generate additional examples
          * process output into json (inputs, m)
          * deterministic generation mode (temp=0, top_p=1)
        (2) Filter additional examples
        (3) add augmented examples into tasks super set
        """
        train_tasks = os.listdir(train_dir)
        for task in train_tasks:
            D_task = load_dataset("json", data_files=f"{train_dir}/{task}")
            k_idx = np.random.randint(low=0, high=len(D_task), size=k).tolist()
            D_subset = D_task.select(k_idx)
        pass

    def to_prompt(self, data):
        """
        data: row from dataset
        """
        pass

    def make_request(self):
        pass

    def _filter(self):
        pass

    def _field_check(self):
        pass

    def _rougel_check(self):
        pass


# k = 4
# g = GenerationPipeline("./bigbench_metadata/mc_tasks.csv", "training_data", 8, "hello")
# g.augment(k, "training_data")
