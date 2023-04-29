import csv
import json
import ast
import os
import openai
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from datasets import load_dataset, Dataset

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

    def augment(self, k: int, train_dir: str, total_qs: int) -> None:
        """
        TODO (unfinished)
        for all train tasks:
        (0) Sample k examples from task
          * format k examples into prompt
        (1) GPT to generate additional examples
          * process output into json (inputs, m)
          * deterministic generation mode (temp=0, top_p=1)
        (2) Filter additional examples
        (3) add augmented examples into tasks super set

        k: amount of in-context examples for generation
        train_dir: file path to directory containing train data json files
        total_qs: total number of question to generate (including in-context
        examples)
        """
        train_tasks = os.listdir(train_dir)
        for task in train_tasks:
            D_task = load_dataset("json", data_files=f"{train_dir}/{task}")
            k_idx = np.random.randint(low=0, high=len(D_task), size=k).tolist()
            D_subset = D_task.select(k_idx)
            prompt = self.to_prompt(D_subset, total_qs)
            # make request
            msg_content = self.make_request(prompt)
            # process request
            self.process_request(msg_content, task)
        pass

    def to_prompt(self, dataset: Dataset, total_qs: int) -> str:
        """
        TODO (review, add task description to prompt)
        dataset: subset to embed as context for prompt
        total_qs: total questions to generate including embedded questions
        """
        keys = ["inputs", "targets", "multiple_choice_targets"]
        # each key has same amount of values want to map each value of each key with each other
        ds_dict = dataset.to_dict()
        n_examples = len(ds_dict["inputs"])

        examples = []
        for i in range(n_examples):
            example = {}
            for key in keys:
                example[key] = ds_dict[key][i]
            examples.append(example)

        prompt = f"Generate a series of {total_qs} questions in a valid JSON format:"
        # add description of task? 
        for i, e in enumerate(examples):
            prompt += f"\nQuestion {i}:\n{e}"
        prompt += f"\nQuestion {n_examples}:"

        return prompt

    def make_request(self, prompt: str, model: str = "gpt-4") -> str:
        """
        Assumes only sending one prompt (e.g., not sending list of prompts)

        prompt: includes in-context examples
        model: model used for generation
        """
        msg_content = ""
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": prompt}],  # change role??
                temperature=0,
            )
            msg_content = completion['choices'][0]['message']['content']
        except openai.error.OpenAIError as e:
            print(f"OpenAIError: {e}.")

        return msg_content

    def process_request(self, msg_content:str, task:str) -> None:
        """
        TODO (review, add filtering)
        processes msg_content from GPT
        (1) converts msg_content to list of dictionaries 
        (2) applying filters to dictionary examples
        (3) append list of dictionaries to json (filename: "{task}_generated.json")

        res: message content from GPT
        task: name of task generated questions correspond to
        """
        generated_examples = self._generation_to_dicts(msg_content)
        # apply filters here
        with open(f"{task}_generated.json", "a") as f:
          json.dump(generated_examples, f)

    def _generation_to_dicts(self, msg_content:str) -> List[Dict]:
      """
      TODO (review)
      returns list of dictionary representing generated questions
      """
      generate_qs = msg_content.split("Question") # change split word?
      generated_json_examples = []
      for i, q in enumerate(generate_qs):
        if i == 0: # end of prompt = "... Question i:"
            msg_dict = ast.literal_eval(q)
        else: # cut off Question:\n that comes before the {<json_question>}
            q = q[q.find('\n'):]
            q = q.rstrip("\n")
            msg_dict = ast.literal_eval(q)
      generated_json_examples.append(msg_dict)

    def _filter(self):
        pass

    def _field_check(self):
        pass

    def _rougel_check(self):
        pass


# k = 4
# g = GenerationPipeline("./bigbench_metadata/mc_tasks.csv", "training_data", 8, "hello")
# g.augment(k, "training_data")
