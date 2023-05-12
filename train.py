import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from src.data import load_datasets, preprocess_dataset
from src.args import create_args
from src.model import ICLModel
from src.trainer import ICLTrainer
from config import TRAIN_TASKS, TEST_TASKS


def train(args):
    # setup models and peft
    base_model = AutoModelForCausalLM.from_pretrained(args.model_str)
    tokenizer = AutoTokenizer.from_pretrained(args.model_str)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    base_model = get_peft_model(base_model, peft_config)
    base_model.print_trainable_parameters()

    model = ICLModel(base_model, k_examples=args.k)

    # setup datasets and dataloaders
    preprocess_fn = lambda ds: preprocess_dataset(
        ds,
        tokenizer,
        method=args.data_method,
        include_choices=args.include_choices,
        num_procs=args.preprocessing_num_workers,
    )
    train_datasets = load_datasets(
        TRAIN_TASKS,
        use_augmented=not args.no_augment,
        preprocess_fn=preprocess_fn,
    )
    val_datasets = load_datasets(
        TEST_TASKS, use_augmented=False, preprocess_fn=preprocess_fn
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to=args.report_to,
        seed=args.seed,
        logging_dir="./logs",
        evaluation_strategy="epoch",
        save_total_limit=5,
        remove_unused_columns=False,
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    trainer = ICLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets,
        eval_dataset=val_datasets,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if args.train:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model()
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if args.evaluate:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if args.push_to_hub:
        trainer.push_to_hub()
        trainer.create_model_card()

    return trainer


if __name__ == "__main__":
    args = create_args()
    train(args)
