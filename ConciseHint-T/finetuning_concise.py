import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model,  PromptTuningConfig, PeftModelForCausalLM
import os
# import swanlab 
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from dataclasses import dataclass, field

# os.environ["SWANLAB_PROJECT"]="qwen3-sft-medical"
MAX_LENGTH = 32768

# swanlab.config.update({
#     "model": "Qwen/Qwen3-1.7B",
#     # "prompt": PROMPT,
#     "data_max_length": MAX_LENGTH,
#     })


def process_func(example):
    input_ids, attention_mask, labels = [], [], []


    instruction = tokenizer(
        f"<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output'].strip()}"+ tokenizer.eos_token+'\n', add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"]
    )

    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:  
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "length_level": example['length_level']}   


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


@dataclass
class ConciseTuningConfig(PromptTuningConfig):
    use_concise_tuning: bool = field(
        default=True,
    )


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", device_map="auto", torch_dtype=torch.bfloat16)

    init_prompt = "-<|im_start|>user\nmake answer concise!<|im_end|>-"
    config = ConciseTuningConfig(
        task_type=TaskType.CAUSAL_LM, 
        prompt_tuning_init="TEXT",
        num_virtual_tokens=len(tokenizer.tokenize(init_prompt)),  
        prompt_tuning_init_text=init_prompt,
        tokenizer_name_or_path="Qwen/Qwen3-1.7B",
        use_concise_tuning= True,
    )


    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    from my_peft_model_function import my_peft_model_for_causal_lm_forward
    PeftModelForCausalLM.forward = my_peft_model_for_causal_lm_forward


    # load full dataset
    full_jsonl_path = "MixChain-Z-GSM8K_level-0.jsonl"

    full_df = pd.read_json(full_jsonl_path, lines=True)
    full_ds = Dataset.from_pandas(full_df)
    # split training and validation datasets
    train_test_split = full_ds.train_test_split(test_size=0.01, seed=42)  #
    train_ds = train_test_split['train']
    eval_ds = train_test_split['test']
    # process data
    train_dataset = train_ds.map(process_func, remove_columns=full_ds.column_names)
    eval_dataset = eval_ds.map(process_func, remove_columns=full_ds.column_names)


    args = TrainingArguments(
        output_dir="./output/Qwen3_1.7B-finetuning_concise_level-0",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=1000,
        logging_steps=20,
        # num_train_epochs=3,
        num_train_epochs=6,
        save_steps=1000,
        learning_rate=1e-4,
        # report_to="swanlab",
        run_name="qwen3-1.7B",
        remove_unused_columns = False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    swanlab.finish()