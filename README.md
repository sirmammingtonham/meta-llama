# Meta-LLaMA

## Abstract
We introduce Meta-LLaMA, a novel framework that combines the techniques of SelfInstruct and Meta-ICL to improve the few-shot learning (FSL) capabilities of large language models (LLMs). LLMs have shown remarkable performance in various natural language processing tasks but still exhibit limitations in FSL, which hinders their generalizability. We attempt to address this problem by meta-training LLMs with augmented data generated using a data generation method inspired by SelfInstruct. Our framework consists of two main stages: data generation and meta-training. We evaluate our framework on the BIG-Bench benchmark, a collection of diverse and challenging tasks, and compare the performance of our meta-trained models with baseline models. Our experimental results demonstrate that Meta-LLaMA slightly improves the few-shot learning performance of billion parameter LLMs, however they also suggest that the unique and varied characteristics exhibited by BIG-bench tasks result in the models acquiring a limited set of transferable skills during meta-training. We believe that our framework offers a promising approach to enhance generate additional data for diverse tasks, and could be a useful tool in constructing large datasets of diverse tasks without significantly less cost.

## Installation
1. `pip install -r requirements.txt`

## Dataset
1. We provide all dataset files in [data](data).

## Data generation
1. Create a file called private.py with `sk = "YOUR_OPENAI_KEY"`
2. `python generate_data.py`

## Training and evaluation
1. `python train.py --train --evaluate`
   * See [src/args.py](src/args.py) for full list of arguments or run `python train.py -h`. 