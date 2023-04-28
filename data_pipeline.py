import csv
import numpy as np
from datasets import load_dataset

np.random.seed(0)

class Pipeline:
    def __init__(self, tasks_dir: str, task_seed_size: int, sk: str) -> None:
        """
        tasks_dir: name of file (csv) of tasks to augment
        task_seed_size: size of seed examples per task
        sk: OpenAI api key
        """
        with open(tasks_dir, newline="") as f:
            reader = csv.reader(f)
            tasks = [row[-1] for row in reader]
        tasks = tasks[1:]
        f.close()

        img_tasks = ["cifar10_classification", "mnist_ascii"]

        self.seed_tasks = {}
        self.missing_tasks = []
        for task in tasks:
            print(f"processing: {task}")
            if task in img_tasks:
                continue
            try:
                D_task = load_dataset("tasksource/bigbench", task, split="train")
                k_idx = np.random.randint(
                    low=0, high=len(D_task), size=task_seed_size
                ).tolist()
                D_init = D_task.select(k_idx)
                self.seed_tasks[task] = D_init
            except:
                print(f"task not found: {task}")
                self.missing_tasks.append(task)

        print(self.seed_tasks.keys())
        print(self.missing_tasks)

    def augment(self):
        """
        for all tasks:
        (0) Sample k examples from task
          * format k examples into prompt
        (1) GPT to generate additional examples
          * process output into json (inputs, m)
        (2) Filter additional examples
        (3) add augmented examples into task set
        """
        pass

    def _filter(self):
        pass


# Pipeline("./bigbench_metadata/mc_tasks.csv", 8, "hello")
