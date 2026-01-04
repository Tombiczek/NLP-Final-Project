import numpy as np
import evaluate
from datasets import DatasetDict
from tqdm import tqdm
import torch
import sys

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import get_peft_model, TaskType, PeftConfig

def _get_adapters():
    """Lazy import adapters library."""
    try:
        import adapters
        from adapters import AdapterTrainer, Seq2SeqAdapterTrainer
        return adapters, AdapterTrainer, Seq2SeqAdapterTrainer
    except ImportError:
        return None, None, None


def _safe_get_peft_model(model, peft_config):
    """
    Wrapper dla get_peft_model obsługujący konflikt z biblioteką adapters.
    Adapters patchuje metodę add_adapter, co powoduje konflikt z PEFT.
    """
    from peft import PeftModel
    from peft.peft_model import (
        PeftModelForSequenceClassification,
        PeftModelForSeq2SeqLM,
        PeftModelForCausalLM,
        PeftModelForTokenClassification,
        PeftModelForQuestionAnswering,
        PeftModelForFeatureExtraction,
    )
    classes_to_check = [
        PeftModel,
        PeftModelForSequenceClassification,
        PeftModelForSeq2SeqLM,
        PeftModelForCausalLM,
        PeftModelForTokenClassification,
        PeftModelForQuestionAnswering,
        PeftModelForFeatureExtraction,
    ]
    
    original_methods = {}
    
    def make_patched_add_adapter(original_method):
        def patched_add_adapter(self, adapter_name, peft_config, low_cpu_mem_usage=False, **kwargs):
            try:
                return original_method(self, adapter_name, peft_config, low_cpu_mem_usage=low_cpu_mem_usage, **kwargs)
            except TypeError as e:
                if "unexpected keyword argument 'low_cpu_mem_usage'" in str(e):
                    return original_method(self, adapter_name, peft_config, **kwargs)
                raise e
        return patched_add_adapter


    for cls in classes_to_check:
        if 'add_adapter' in cls.__dict__:
            original_methods[cls] = cls.add_adapter
            setattr(cls, 'add_adapter', make_patched_add_adapter(cls.add_adapter))
            
    try:
        return get_peft_model(model, peft_config)
    finally:
        for cls, original in original_methods.items():
            setattr(cls, 'add_adapter', original)

ACCURACY = evaluate.load("accuracy")
F1 = evaluate.load("f1")
ROUGE = evaluate.load("rouge")


def baseline_bert_ai_ga(
    ds_ai_ga: DatasetDict,
    model_name: str = "bert-base-uncased",
    output_dir: str = "./bert_baseline_output",
    num_epochs: int = 1,
    batch_size: int = 16,
    max_length: int = 256,
    learning_rate: float = 2e-5,
):
    """
    Trenuje BERT na zbiorze AI-GA i zwraca accuracy oraz F1 (macro).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(batch):
        return tokenizer(batch["abstract"], truncation=True, max_length=max_length)

    tokenized = ds_ai_ga.map(preprocess, batched=True, remove_columns=["title", "abstract"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": ACCURACY.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": F1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer.evaluate(tokenized["test"])


def baseline_mt5_zero_shot(
    ds: DatasetDict,
    model_name: str = "google/mt5-small",
    num_samples: int = 200,
    max_input_length: int = 512,
    max_output_length: int = 64,
    batch_size: int = 8,
):
    """
    Zero-shot summarization z mT5. Zwraca metryki ROUGE.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    ds_sample = ds["test"].select(range(min(num_samples, len(ds["test"]))))

    predictions = []
    references = []

    for i in tqdm(range(0, len(ds_sample), batch_size), desc="Generating summaries"):
        batch = ds_sample[i : i + batch_size]
        
        inputs = tokenizer(
            batch["text"],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_input_length,
        ).to(device)

        outputs = model.generate(**inputs, max_length=max_output_length, num_beams=4)
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
        references.extend(batch["summary"])

    return ROUGE.compute(predictions=predictions, references=references)


def train_bert(
    ds: DatasetDict,
    model_name: str = "bert-base-uncased",
    output_dir: str = "./bert_output",
    peft_config: PeftConfig = None,
    adapter_config: str = None,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
):
    """
    Uniwersalna funkcja trenująca BERT (Full Fine-Tuning, PEFT lub Adapters).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(batch):
        return tokenizer(batch["abstract"], truncation=True, max_length=max_length)

    tokenized = ds.map(preprocess, batched=True, remove_columns=["title", "abstract"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    if peft_config is not None:
        model = _safe_get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        trainer_cls = Trainer
    elif adapter_config is not None:
        adapters, AdapterTrainer, _ = _get_adapters()
        if adapters is None:
            raise ImportError("Biblioteka 'adapters' nie jest zainstalowana.")
        adapters.init(model)
        model.add_adapter("default", config=adapter_config)
        model.train_adapter("default")
        trainer_cls = AdapterTrainer
    else:
        trainer_cls = Trainer
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": ACCURACY.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": F1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer.evaluate(tokenized["test"])


def train_mt5(
    ds: DatasetDict,
    model_name: str = "google/mt5-small",
    output_dir: str = "./mt5_output",
    peft_config: PeftConfig = None,
    adapter_config: str = None,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    max_input_length: int = 512,
    max_target_length: int = 64,
):
    """
    Uniwersalna funkcja trenująca mT5 (Full Fine-Tuning, PEFT lub Adapters).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if peft_config is not None:
        if hasattr(peft_config, 'peft_type') and str(peft_config.peft_type) == "PeftType.PREFIX_TUNING":
            if model.config.model_type in ["t5", "mt5"]:
                if hasattr(model.config, "d_kv") and hasattr(model.config, "num_heads"):
                    correct_token_dim = model.config.num_heads * model.config.d_kv
                    peft_config.token_dim = correct_token_dim
                
        model = _safe_get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        trainer_cls = Seq2SeqTrainer
    elif adapter_config is not None:
        adapters, _, Seq2SeqAdapterTrainer = _get_adapters()
        if adapters is None:
            raise ImportError("Biblioteka 'adapters' nie jest zainstalowana.")
        adapters.init(model)
        model.add_adapter("default", config=adapter_config)
        model.train_adapter("default")
        trainer_cls = Seq2SeqAdapterTrainer
    else:
        trainer_cls = Seq2SeqTrainer

    def preprocess_function(examples):
        inputs = examples["text"]
        targets = examples["summary"]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(preprocess_function, batched=True, remove_columns=["text", "summary", "id", "url", "title"])

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = predictions.astype(np.int32)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = labels.astype(np.int32)
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = ROUGE.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        report_to="wandb",
        predict_with_generate=True,
        generation_max_length=max_target_length,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
    )

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer.evaluate(tokenized["test"])