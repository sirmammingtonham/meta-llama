import csv
import json
import ast
import os
import openai
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from rouge_score import rouge_scorer
from typing import List, Dict, Tuple
from datasets import load_dataset, concatenate_datasets, Dataset
import time

TASK_SELECT_SEED = 10
np.random.seed(0)

EXCLUDED_TASKS = ["cifar10_classification", "mnist_ascii"]


class GenerationPipeline:
    def __init__(
        self,
        tasks_dir: str,
        train_dir: str,
        train_dir_gen: str,
        error_dir: str,
        task_seed_size: int,
        sk: str,
        filter_errors: str = "filter_errors.csv",
        error_file: str = "errors.txt",
    ) -> None:
        """
        tasks_dir: name of file (csv) of tasks to augment
        train_dir: file path to directory to write data to
        train_dir_gen: file path to dump generated instances
        error_dir: file path to dump error related data
        task_seed_size: size of seed examples per task
        filter_errors: file name (csv) to write filter errors to
        sk: OpenAI api key
        """
        openai.api_key = sk
        self.train_dir = train_dir
        self.train_dir_gen = train_dir_gen
        self.error_dir = error_dir
        self.filter_errors = filter_errors
        self.error_file = error_file

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

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        # # initialize seed task jsons in training directory
        self.missing_tasks = self._init_task_seeds(task_seed_size, train_dir)
        self.train_tasks = filter(
            lambda x: x not in self.missing_tasks, self.train_tasks
        )

        # initialize json files to dump generated examples into
        if not os.path.exists(self.train_dir_gen):
            os.makedirs(self.train_dir_gen)

        # initialize error recording directory
        if not os.path.exists(self.error_dir):
            os.makedirs(self.error_dir)

        for task in self.train_tasks:
            with open(f"{self.train_dir_gen}/{task}.json", "w") as f:
                json.dump([], f)
            f.close()

        self.descr_df = pd.read_csv("./bigbench_metadata/task_descriptions.csv")

    def _init_task_seeds(self, task_seed_size: int, train_dir: str) -> List[str]:
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
                is_generated = [False] * len(D_init)
                D_init = D_init.add_column("is_generated", is_generated)
                D_init = D_init.add_column("true_idx", k_idx)
                D_init.remove_columns("idx")
                D_init.to_json(path, orient="records", lines=False)
            except:
                missing_tasks.append(task)
        return missing_tasks

    def augment(self, k: int, total_qs: int) -> None:
        """
        for all train tasks:
        (0) Sample k examples from task
          * format k examples into prompt
        (1) GPT to generate additional examples
          * process output into json (inputs, m)
          * deterministic generation mode (temp=0, top_p=1)
        (2) Filter additional examples
        (3) add augmented examples into tasks super set

        k: amount of in-context examples for generation
        total_qs: total number of question to generate (including in-context
        examples)
        """
        train_tasks = os.listdir(self.train_dir)
        for i, task in enumerate(train_tasks):
            task = task[:-5]  # get rid of .json
            D_task = load_dataset(
                "json", data_files=f"{self.train_dir}/{task}.json", split="train"
            )
            D_task_aug = load_dataset(
                "json", data_files=f"{self.train_dir_gen}/{task}.json", split="train"
            )
            # sample indeces for selection, treat len(D_task) as offset
            k_idx = np.random.randint(
                low=0, high=len(D_task) + len(D_task_aug), size=k
            ).tolist()
            k_idx_real = list(filter(lambda x: x < len(D_task), k_idx))
            k_idx_aug = list(
                np.array(list(filter(lambda x: x >= len(D_task), k_idx))) - len(D_task)
            )
            # select examples from both real and augmented task dataset
            D_subset = D_task.select(k_idx_real)
            D_subset_aug = D_task_aug.select(k_idx_aug)
            D_examples = concatenate_datasets([D_subset, D_subset_aug], axis=0)
            D_examples = D_examples.shuffle(seed=TASK_SELECT_SEED)
            # construct prompt
            prompt = self.to_prompt(D_examples, total_qs, task)
            # make request
            msg_content = self.make_request(prompt)
            # process request
            filter_summary = self.process_request(msg_content, task)
            # record error summary
            with open(f"{self.error_dir}/{self.filter_errors}", "a+") as f:
                # task, parse filter, rl filter, fc filter <-- columns
                filter_summary["task"] = task
                dict_writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "task",
                        "parse_filters",
                        "field_check_filters",
                        # "rougel_filters",
                    ],
                )
                if i == 0:
                    dict_writer.writeheader()
                dict_writer.writerow(filter_summary)
            f.close()

    def to_prompt(self, dataset: Dataset, total_qs: int, task: str) -> str:
        """
        dataset: subset to embed as context for prompt
        total_qs: total questions to generate including embedded questions
        task: task name
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

        prompt = """Generate a series of 10 diverse questions related to TASK DESCRIPTION in a JSON format with the following schema:
```
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "object",
  "properties": {
    "inputs": {
      "type": "string"
    },
    "targets": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "multiple_choice_targets": {
      "type": "array",
      "items": {
        "type": "string"
      }
    }
  },
  "required": ["inputs", "targets", "multiple_choice_targets"]
}
```"""
        task_descr = self.descr_df[(self.descr_df["task"] == task)][
            "description"
        ].values[0]
        prompt += f"\nTASK DESCRIPTION: {task_descr}"

        for i, e in enumerate(examples):
            prompt += f"\nQuestion {i}:\n{e}"
        prompt += f"\nQuestion {n_examples}:"

        return prompt

    def make_request(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """
        Assumes only sending one prompt (e.g., not sending list of prompts)

        prompt: includes in-context examples
        model: model used for generation
        """
        msg_content = ""
        try:
            start = time.time()
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                ],
                temperature=0,
                presence_penalty=1,
            )
            end = time.time()
            print("REQUEST TIME:", end - start)
            msg_content = completion["choices"][0]["message"]["content"]
        except openai.error.OpenAIError as e:
            print(f"OpenAIError: {e}.")

        return msg_content

    def process_request(self, msg_content: str, task: str) -> Dict:
        """
        processes msg_content from GPT
        (1) parse msg_content to python dict
        (2) applying filters to dictionary examples
        (4) write filtered messages to appropriate task db (json)

        res: message content from GPT
        task: name of task generated questions correspond to
        """
        questions_parsed, parsed_errs = self._generation_to_dicts(msg_content)
        added_questions, fc_errs = self._field_check(questions_parsed)
        # added_questions, rl_errs = self._rougel_check(task, questions_fc_pass)

        filter_summary = {
            "parse_filters": parsed_errs,
            "field_check_filters": fc_errs,
            # "rougel_filters": rl_errs,
        }

        with open(f"{self.train_dir_gen}/{task}.json", "r") as f:
            task_ds = json.load(f)
        f.close()

        task_ds.extend(added_questions)

        with open(f"{self.train_dir_gen}/{task}.json", "w") as f:
            json.dump(task_ds, f)
        f.close()

        return filter_summary

    def _generation_to_dicts(self, msg_content: str) -> Tuple[List[Dict], int]:
        """
        returns list of dictionary representing generated questions
        """
        # change split word? what if 'Question: \d' in examples
        generate_qs = re.split("Question \d:", msg_content)
        generated_dict_examples, errs = [], 0
        for i, q in enumerate(generate_qs):
            if i == 0:  # end of prompt = "... Question i:"
                try:
                    msg_dict = ast.literal_eval(q)
                except:
                    print(f"unable to convert to dict: {q}")
                    self.dump_error(q, self.error_file)
                    errs += 1
            else:  # cut off Question:\n that comes before the {<json_question>}
                q = q[q.find("\n") :]
                q = q.rstrip("\n")
                try:
                    msg_dict = ast.literal_eval(q)
                except:
                    print(f"unable to convert to dict: {q}")
                    self.dump_error(q, self.error_file)
                    errs += 1
            generated_dict_examples.append(msg_dict)

        return generated_dict_examples, errs

    def _field_check(self, questions: List[Dict]) -> Tuple[List[Dict], int]:
        """
        questions: generated question list in dict format
        """
        keys = set(["inputs", "targets", "multiple_choice_targets"])
        out, errs = [], 0
        for q in questions:
            if len(keys.difference(set(q.keys()))) == 0:
                out.append(q)
            else:
                self.dump_error(q, self.error_file)
                errs += 1
        return out, errs

    def _rougel_check(self, task: str, questions: List[Dict]) -> Tuple[List[Dict], int]:
        """
        assumes questions pass _field_check
        task: task name
        questions: generated question list in dict format (e.g.,
        {inputs:<str>, targets:<str>, multiple_choice_targets:<str>})
        """
        # get current examples in db
        with open(f"{self.train_dir}/{task}.json", "r") as f:
            ds_init = json.load(f)
        f.close()
        with open(f"{self.train_dir_gen}/{task}.json", "r") as f:
            existing_generated_questions = json.load(f)
        f.close()

        ds_init.extend(existing_generated_questions)

        ds_inputs = [question["inputs"] for question in ds_init]
        generated_inputs = [question["inputs"] for question in questions]

        scorer = rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=False
        )  # same as self-instruct setting
        scores, excluded_qs, errs = defaultdict(list), set(), 0
        for i, existing_q in enumerate(ds_inputs):
            for j, new_q in enumerate(generated_inputs):
                score = scorer.score(new_q, existing_q)["rougeL"].fmeasure
                scores[j].append((i, score))
                if score > 0.7:
                    self.dump_error(existing_q, self.error_file)
                    excluded_qs.add(j)
                    errs += 1

        return [q for i, q in enumerate(questions) if i not in excluded_qs], errs

    def dump_error(self, res: str, error_file: str):
        with open(f"{self.error_dir}/{error_file}", "a") as f:
            f.write(f"{res}\n")
