import torch
import numpy as np
import evaluate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EvalPrediction,
)
from peft import get_peft_model, prepare_model_for_int8_training, LoraConfig, TaskType
from src.data import load_datasets, preprocess_dataset
from src.args import create_args
from src.model import ICLModel
from src.trainer import ICLTrainer
from config import TRAIN_TASKS, TEST_TASKS


def train(args):
    # setup models and peft
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_str,  load_in_8bit=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_str)
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "v_proj",
        ],
    )
    base_model = prepare_model_for_int8_training(base_model)
    base_model = get_peft_model(base_model, peft_config)
    base_model.print_trainable_parameters()

    model = ICLModel(base_model, k_examples=args.k)

    # setup datasets and dataloaders
    assert (
        args.data_method == "direct"
    ), "we have not implemented evaluation for channel method yet!"
    preprocess_fn = lambda ds: preprocess_dataset(
        ds,
        tokenizer,
        method=args.data_method,
        num_procs=args.preprocessing_num_workers,
    )
    train_datasets = load_datasets(
        TRAIN_TASKS,
        data_dir=args.train_dir,
        use_augmented=not args.no_augment,
        preprocess_fn=preprocess_fn,
    )
    val_datasets = load_datasets(
        TEST_TASKS,
        data_dir=args.test_dir,
        use_augmented=False,
        preprocess_fn=preprocess_fn,
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
        # ddp_backend="gloo",
        save_total_limit=5,
        fp16=True,
    )

    metric = evaluate.load("accuracy")

    def preprocess_logits_for_metrics(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(logits, tuple):
            logits = logits[0]
        label_mask = torch.where(
            labels != 0, torch.ones_like(labels), torch.zeros_like(labels)
        )
        return logits.argmax(dim=-1) * label_mask

    def compute_metrics(eval_preds: EvalPrediction) -> dict:
        preds, labels = eval_preds

        # shift for autoregressive
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)

        # select only the non padding ones
        preds = preds[(preds != -100) & (preds != 0)]
        labels = labels[(labels != -100) & (labels != 0)]
        return metric.compute(predictions=preds, references=labels)

    trainer = ICLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets,
        eval_dataset=val_datasets,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if args.train:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model()
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if args.evaluate:
        metrics = trainer.predict()
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    if args.push_to_hub:
        trainer.push_to_hub()
        trainer.create_model_card()

    return trainer


if __name__ == "__main__":
    args = create_args()
    train(args)
