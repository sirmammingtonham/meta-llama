from generation_args import create_args
from private import sk
from generation_pipeline import GenerationPipeline
from config import ETHAN_TASKS

if __name__ == "__main__":
    gp = GenerationPipeline(
        "./bigbench_metadata/mc_tasks.csv",
        "training_data",
        "training_data_generated",
        "errors",
        50,
        sk,
    )
    k, total_qs = 4, 10
    gp.augment(k, total_qs, ETHAN_TASKS)
