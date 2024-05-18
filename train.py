import argparse
import torch
import pandas as pd
import numpy as np
from functools import partial
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq, 
    set_seed,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from nltk.tokenize import sent_tokenize
from prompt import input_extract_all, target_extract, prompt_dict_brief, prompt_dict_instructs
from metrics import compute_overall_score

def preprocess_function(text, max_input_length, max_target_length):
    model_inputs = tokenizer(
        text["input"],
        max_length=max_input_length,
        truncation=True,
        add_special_tokens=True,
        padding="max_length",
        return_tensors="pt",
    )
    labels = tokenizer(
        text["target"], 
        max_length=max_target_length, 
        truncation=True, 
        add_special_tokens=True, 
        padding="max_length", 
        return_tensors="pt"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.where(predictions != -100, predictions, tokenizer.sep_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    
    results = compute_overall_score(decoded_labels, decoded_preds, metrics=["bleu", "rouge", "bertscore", "meteor", "align", "medcon"])
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="luqh/ClinicalT5-base", choices=["luqh/ClinicalT5-base", "luqh/ClinicalT5-large"])
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max_input_length", type=int, default=1596)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--target", type=str, default="discharge_instructions", choices=["discharge_instructions", "brief_hospital_course"])

    args = parser.parse_args()
    model_name = args.model_name
    batch_size = args.batch_size
    epochs = args.epochs
    max_input_length = args.max_input_length
    fold = args.fold
    target = args.target

    if target == "brief_hospital_course":
        prompt_for_input = prompt_dict_brief
        max_target_length = 832

    elif target == "discharge_instructions":
        prompt_for_input = prompt_dict_instructs
        max_target_length = 792

    raw_data_file = ""

    df_5folds = pd.read_csv(raw_data_file, compression='gzip', header=0, sep=',', quotechar='"')
    df_fold_train = df_5folds[df_5folds["fold"] != fold]
    df_fold_valid = df_5folds[df_5folds["fold"] == fold]

    input_extracted_with_prompt = df_fold_train["text_without_target"].apply(lambda x: input_extract_all(x, prompt_for_input))
    target_extracted_text = df_fold_train[target].apply(target_extract)

    input_extracted_with_prompt_valid = df_fold_valid["text_without_target"].apply(lambda x: input_extract_all(x, prompt_for_input))
    target_extracted_text_valid = df_fold_valid[target].apply(target_extract)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, sep_token="<sep>")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, from_flax=True, torch_dtype=torch.float16)

    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q", "v"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="SEQ_2_SEQ_LM", 
        use_rslora=True
    )
    
    peft_model = get_peft_model(model, lora_config)

    train_dataset = Dataset.from_dict({
        "input": input_extracted_with_prompt,
        "target": target_extracted_text
    })
    val_dataset = Dataset.from_dict({
        "input": input_extracted_with_prompt_valid,
        "target": target_extracted_text_valid
    })

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    prepared_preprocess_function = partial(preprocess_function, max_input_length=max_input_length, max_target_length=max_target_length)

    tokenized_datasets = train_dataset.map(prepared_preprocess_function, batched=True)
    tokenized_datasets_val = val_dataset.map(prepared_preprocess_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(train_dataset.column_names)
    tokenized_datasets_val = tokenized_datasets_val.remove_columns(val_dataset.column_names)

    args = Seq2SeqTrainingArguments(
        output_dir=f"{model_name.split('/')[-1]}-finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        learning_rate=1e-4,
        weight_decay=0.01,
        save_total_limit=6,
        predict_with_generate=True,
        generation_max_length=max_target_length,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=peft_model,
        args=args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    set_seed(42)
    train_result = trainer.train()
    trainer.save_model()
    torch.cuda.empty_cache()