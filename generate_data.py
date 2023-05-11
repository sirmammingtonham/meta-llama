from src.generation.generation_args import create_args
from src.generation.generation_pipeline import GenerationPipeline
from config import ETHAN_TASKS, SUNIL_TASKS
from private import sk

if __name__ == "__main__":
    gp = GenerationPipeline(
        tasks_dir="data/metadata/mc_tasks.csv",
        train_dir="data/training_data",
        train_dir_gen="data/augment_train_v2",
        error_dir="errors_v2",
        task_seed_size=50,
        sk=sk,
        filter_errors="filter_errors.csv",
        error_file="errors.txt",
        epochs=40,
        max_generations=200,
    )
    k, total_qs = 4, 10
    gp.augment(
        k,
        total_qs,
        SKIP_TASKS=[],
        TARG_TASKS=[],
    )
