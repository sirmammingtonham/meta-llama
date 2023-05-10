from generation.generation_args import create_args
from private import sk
from generation.generation_pipeline import GenerationPipeline
from config import SUNIL_TASKS

if __name__ == "__main__":
    gp = GenerationPipeline(
        "bigbench_metadata/mc_tasks.csv",
        "training_data",
        "training_data_generated_v2",
        "errors_v2",
        50,
        sk,
    )
    k, total_qs = 4, 10
    gp.augment(k, total_qs, SKIP_TASKS= ['kanji_ascii','parsinlu_qa'],TARG_TASKS=SUNIL_TASKS)
