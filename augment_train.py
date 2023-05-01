from generation_args import create_args
from generation_pipeline import GenerationPipeline


if __name__ == "__main__":
    args = create_args()
    tasks_dir = args.tasks_dir
    train_dir = args.train_dir
    train_dir_gen = args.train_dir_gen
    error_dir = args.error_dir
    task_seed_size = args.task_seed_size
    filter_errors = args.filter_errors
    sk = args.sk
    k = args.k
    total_qs = args.total_qs

    gp = GenerationPipeline(
        tasks_dir,
        train_dir,
        train_dir_gen,
        error_dir,
        task_seed_size,
        filter_errors,
        sk,
    )

    gp.augment(k, total_qs)
