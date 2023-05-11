from src.generation.generation_utils import init_test_data, init_clean_train_data

from config import TRAIN_TASKS, TEST_TASKS, IGNORED_TASKS

if __name__ == "__main__":
    missing_train = init_clean_train_data(TRAIN_TASKS, IGNORED_TASKS)
    missing_test = init_test_data(TEST_TASKS, IGNORED_TASKS)
    print(
        f"MISSING: \n\ntrain: {missing_train},\n\n {'-' *50} \n\ntest: {missing_test}"
    )
