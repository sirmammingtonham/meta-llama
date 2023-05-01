import argparse


def create_args() -> argparse.ArgumentParser:
    """Defines a parameter parser for all of the arguments of the application."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tasks_dir",
        type=str,
        help="name of file (csv) of tasks to augment",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        help="file path to directory to write data to",
    )
    parser.add_argument(
        "--train_dir_gen",
        type=str,
        help="file path to dump generated instances",
    )
    parser.add_argument(
        "--error_dir",
        type=str,
        help="file path to dump error related data",
    )
    parser.add_argument(
        "--task_seed_size",
        type=int,
        help="size of seed examples per task",
    )
    parser.add_argument(
        "--filter_errors",
        default="filter_errors.csv",
        type=str,
        help="file name (csv) to write filter errors to",
    )
    parser.add_argument(
        "--sk",
        type=str,
        help="OpenAI api key",
    )

    parser.add_argument(
        "--k",
        type=int,
        help="amount of in-context examples for generation"
        )

    parser.add_argument(
        "--total_qs",
        type=int,
        help="total number of question to generate (including in-context examples)"
        )
   
    return parser.parse_args()
