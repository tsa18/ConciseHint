import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoTokenizer, Qwen3ForCausalLM
from peft import LoraConfig, TaskType, get_peft_model,  PromptTuningConfig
import os
from dataclasses import dataclass, field
from data_utils import load_dataset
from logger import setup_logger
import argparse
from tqdm import trange
import logging
from pytorch_lightning import seed_everything
from peft import AutoPeftModelForCausalLM


import tiktoken
# measure tge token usage 
def token_measure_o1(text):
    tokenizer = tiktoken.encoding_for_model("o1")
    return len(tokenizer.encode(text, disallowed_special=()))

def generate_custom_left_padding_attention_mask(input_ids: torch.Tensor, pad_token_id: int, eos_token_id: int) -> torch.Tensor:
    """
    生成自定义的attention_mask，遵循以下规则：
    1. 默认情况下，pad_token_id 和 eos_token_id 对应的mask为0。
    2. 从左到右扫描，一旦遇到第一个既非pad_token_id也非eos_token_id的token，
    该token及其之后的所有token（包括后续的pad_token_id和eos_token_id）的mask都置为1。

    Args:
        input_ids (torch.Tensor): 输入的token ID张量，形状为 (batch_size, sequence_length)。
        pad_token_id (int): 填充token的ID。
        eos_token_id (int): 序列结束token的ID。

    Returns:
        torch.Tensor: 生成的attention mask张量，形状与input_ids相同，dtype为torch.int。
    """
    batch_size, seq_len = input_ids.shape

    # 1. 识别“内容”token（即不是pad也不是eos的token）
    # 这会生成一个布尔张量，形状与 input_ids 相同
    # 例如：[0, 0, 0, 1, 1, 1, 0] (A,B,C是内容，PAD是0)
    content_indicators = (input_ids != pad_token_id) & (input_ids != eos_token_id)

    # 2. 找到每个序列中第一个“内容”token的索引
    # torch.argmax 在布尔张量上工作时，会返回第一个 True (1) 的索引。
    # 如果一行全是 False (即没有内容token，全是pad/eos)，argmax会返回0。
    first_content_indices = torch.argmax(content_indicators.int(), dim=1) # shape: (batch_size,)

    # 3. 处理没有“内容”token的序列
    # 对于那些完全由pad/eos组成的序列（`content_indicators.any(dim=1)` 为 False），
    # 它们的 `first_content_indices` 仍然是 0。我们希望它们的mask全是0。
    # 解决方法是将其 `first_content_indices` 设置为 `seq_len`，这样后续的 `>=` 比较就会全部为 False。
    has_content = content_indicators.any(dim=1) # shape: (batch_size,)
    first_content_indices[~has_content] = seq_len

    # 4. 基于 `first_content_indices` 构建最终的 attention mask
    # 创建一个与序列长度相同的索引范围张量，形状为 (1, seq_len)
    col_indices = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

    # 将 first_content_indices 扩展为 (batch_size, 1)，以便与 col_indices 进行广播比较
    first_content_indices_expanded = first_content_indices.unsqueeze(1)

    # 比较：如果列索引 (j) 大于等于该行第一个内容token的索引 (i)，则为1，否则为0
    # 结果张量形状为 (batch_size, seq_len)
    attention_mask = (col_indices >= first_content_indices_expanded).int()
    
    return attention_mask


@dataclass
class ConciseTuningConfig(PromptTuningConfig):
    use_concise_tuning: bool = field(
        default=True,
    )
    fixed_interval: int = field(
        default=128,
    )

def parse_args():
    parser = argparse.ArgumentParser(description="ConciseHint Experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to use (gsm8k, aime24, gpqa_diamond)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Name or path of the model to use",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum number of tokens per request",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.95, help="Top p for sampling")
    parser.add_argument(
        "--bs", type=int, default=1, help="batch size to process questions"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exp", type=str, default="exp", help="annotation for this exp")
    parser.add_argument("--repeat_exp_num", default=5, type=int, help="The number of repetitive exps")

    return parser.parse_args()


from evaluator import Evaluator
from grader import grade_answer
# eval if the answer is correct
def eval_results(dataset, results, targets):
    correct_num = 0
    for result, target in zip(results, targets):
        result = result.replace('\\boxed{}','') # remove \\boxed{} in input prompt
        if dataset == "gsm8k" or dataset == "gsm8k_dev":
            target = Evaluator.find_answer_gsm8k(target)
        if dataset=="gsm8k" or dataset == "gsm8k_dev" or dataset=="aime24":
            correct = (target==Evaluator.extract_predicted_answer(result)) or (r"\boxed{"+str(target)+r"}" in result) \
                    or grade_answer(target,  Evaluator.extract_predicted_answer(result))
            extracted_last_k = Evaluator.extract_predicted_answer(result, last=3) 
            correct = correct or (isinstance(extracted_last_k, tuple) and any(grade_answer(target, answer) for answer in extracted_last_k)) # corner cases where \boxed is not in answer and the last number is also not the answer
        elif dataset=="gpqa_diamond":

            def find_last_answer(response, strings):
                indices = [response.rfind(string) for string in strings]
                return response[max(indices):]

            answer_patterns = [
                "The correct answer is", r"\boxed{", 
                "the correct answer is", "the correct answer would be", "Final Answer:", "the answer is"
            ]
            result = find_last_answer(result, answer_patterns)
            correct = f"The correct answer is {target}" in result or f"The correct answer is **{target}**" in result or f"The correct answer is **{target}" in result \
                       or f"The correct answer is <{target}>" in result  or f"The correct answer is <{target}" in result or r"\boxed{"+str(target) in result \
                       or r"\boxed{\text{"+str(target) in result \
                       or f"the correct answer is {target}"  in result \
                       or f"**The correct answer is <Your choice among A, B, C, and D> {target}." in result \
                       or f"the correct answer is option {target}" in result or f"the correct answer would be option {target}" in result \
                       or f"**Final Answer:** {target}" in result or f"the correct answer would be {target}" in result
                       
        else:
            raise NotImplementedError
        correct_num += correct
        if not correct:
            print('-'*50)
            print('not correct start')
            print("result:",result)
            print("target:",target)
            print('-'*50)
                
    accuracy = correct_num/len(results)
    return accuracy

def execute_questions_batch(
    args,
    model,
    questions_batch,
    top_p=0.95,
    temperature=0.6,
    max_token_length = 16384, 
):

    logger = logging.getLogger(name="my_logger")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    total_tokens_num = 0
    total_tokens_num_o1 = 0
    device = "cuda"
    if args.dataset=="gpqa_diamond":
        prepared_prompts = [  f"<|im_start|>user\n{prompt}" + "\nGive your final answer in the format: The correct answer is <Your choice among A, B, C, and D>. For example: The correct answer is C.<|im_end|>" + "\n<|im_start|>assistant\n" + '<think>' for prompt in questions_batch]
        # prepared_prompts = [  f"<|im_start|>user\n{prompt}" + "\nPlease show your choice in the answer field with only the choice letter, e.g., answer: C.<|im_end|>" + "\n<|im_start|>assistant\n" + '<think>' for prompt in questions_batch]
        
    else:
        prepared_prompts = [  f"<|im_start|>user\n{prompt}" + "\nPut your final answer within \\boxed{}.<|im_end|>" + "\n<|im_start|>assistant\n" + '<think>' for prompt in questions_batch]
    print(tokenizer.pad_token)
    model_inputs = tokenizer(prepared_prompts,  padding=True, return_tensors="pt").to(device)
    input_ids =  model_inputs.input_ids
    base_len = 128
    model.peft_config['default'].fixed_interval = base_len
    model.peft_config['default'].intervals = [base_len]

    results = [None]*len( prepared_prompts)
    indices = list(range(len( prepared_prompts)))
    while True:
        # Generate tokens for current batch
        attention_mask = generate_custom_left_padding_attention_mask(input_ids, tokenizer.pad_token_id, tokenizer.eos_token_id)
        output_ids = model.generate(
            input_ids, 
            max_new_tokens= model.peft_config['default'].intervals[-1], 
            do_sample=True, 
            temperature=temperature, 
            top_p=top_p,
            # top_k=20,
            use_cache=True,
            attention_mask = attention_mask, # necessary 
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id, # batch generation: pad a sequence if it is terminated.e.g., <eos> <pad> <pad>  <pad>
            # # eos_token_id=tokenizer.eos_token_id, # when specified eos tokens are generated, terminate the generation
        )

        # Split completed and incomplete sequences
        new_input_ids = []
        accessed_indices = []

        for i in range(output_ids.shape[0]):
            # Check if sequence has completed
            if tokenizer.eos_token_id in output_ids[i, input_ids.shape[1]:].tolist() or tokenizer.pad_token_id in output_ids[i, input_ids.shape[1]:].tolist():
                response = tokenizer.decode(output_ids[i], skip_special_tokens=False)  # Save completed sequence
                save_index = indices[i]
                results[save_index] = response.replace(prepared_prompts[save_index], "")
                accessed_indices.append(save_index)
                token_num = len(tokenizer.encode(tokenizer.decode(output_ids[i], skip_special_tokens=True)))
                token_num_o1 = token_measure_o1( tokenizer.decode(output_ids[i], skip_special_tokens=True) )
                total_tokens_num += token_num 
                total_tokens_num_o1 +=   token_num_o1 
                print('='*50)
                print(response)
                print('token num: ', token_num, 'o1 token num: ', token_num_o1)
                print('='*50)
            # Check if reach the maximun token length
            # elif output_ids[i].shape[0]> max_token_length: 
            elif token_measure_o1( tokenizer.decode(output_ids[i], skip_special_tokens=True) ) > max_token_length: 
                logger.info(f"Token length={output_ids[i].shape[0]}, Exceed max len, truncate.")
                logger.info('='*50)
                response = tokenizer.decode(output_ids[i], skip_special_tokens=False)  # Save sequence
                save_index = indices[i]
                results[save_index] = response.replace(prepared_prompts[save_index], "")
                accessed_indices.append(save_index)
                token_num = len(tokenizer.encode(tokenizer.decode(output_ids[i], skip_special_tokens=True)))
                token_num_o1 = token_measure_o1( tokenizer.decode(output_ids[i], skip_special_tokens=True) )
                total_tokens_num +=token_num
                total_tokens_num_o1 +=   token_num_o1 
                logger.info(response)
                # logger.info('='*50)
                print('='*50)
                print(f"Token length={output_ids[i].shape[0]}, Exceed max len, truncate.")
                print(response)
                print('token num: ', token_num, 'o1 token num: ', token_num_o1)
                print('='*50)
            else:
                new_input_ids.append(output_ids[i])    # Retain for next iteration
        
        for k in accessed_indices:
            indices.remove(k)

        if not new_input_ids:  # All sequences completed
            break
        # Prepare next batch
        input_ids = torch.stack(new_input_ids).to(device)
        # Prepare interval
        current_len = sum(model.peft_config['default'].intervals)
        new_interval = min(1024, int( base_len + 0.2*current_len ) )
        model.peft_config['default'].intervals.append(new_interval)


    logger.info(f"total-tokens in batch: {total_tokens_num}",  )
    logger.info('-'*100)
    return results, total_tokens_num, total_tokens_num_o1


if __name__ == '__main__':

    seed_everything(23)
    args = parse_args()
    # load data for inference
    data = load_dataset(args.dataset)
    # load model
    model = AutoPeftModelForCausalLM.from_pretrained(args.model, device_map="cuda", torch_dtype=torch.bfloat16)
    model.eval()
    # replace ori config with concise config
    ori_config =  model.peft_config['default']
    concise_config = ConciseTuningConfig(
        task_type= ori_config.task_type, 
        prompt_tuning_init= ori_config.prompt_tuning_init,
        num_virtual_tokens= ori_config.num_virtual_tokens,  
        prompt_tuning_init_text= ori_config.prompt_tuning_init_text,
        tokenizer_name_or_path=ori_config.tokenizer_name_or_path,
        use_concise_tuning= True,
        fixed_interval = 128
    )
    model.peft_config['default'] = concise_config
    from my_peft_model_function import my_prepare_inputs_for_generation
    PeftModelForCausalLM.prepare_inputs_for_generation= my_prepare_inputs_for_generation

    logger = setup_logger(name="my_logger", log_dir='experiment_logs', \
                          log_filename=f"{args.model.split('/')[-1]}-{args.dataset}-{args.repeat_exp_num}-bs={args.bs}-exp-{args.exp}.log", \
                          log_level=logging.DEBUG) 

    accuracy_across_exps, token_num_across_exps, token_num_o1_across_exps = [], [], []
    for exp_id in range(args.repeat_exp_num):
        logger.info("="*50+f" Start {exp_id}-th experiment "+"="*50)
        results, targets = [], []
        total_token_num, total_token_num_o1 = 0, 0
        bs = args.bs
        for s in trange(0, len(data), bs, desc="Processing questions batch"):
            data_batch = data[s: s+bs]
            questions_batch = [ item["problem"].strip() for item in data_batch]
            target_batch = [ item["answer"].strip() for item in data_batch]
            # generate answers for a batch of questions
            results_batch, total_tokens_num_batch, total_tokens_num_o1_batch  = execute_questions_batch(
                args,
                model,
                questions_batch,
                top_p=args.top_p,
                temperature=args.temperature,
                max_token_length= args.max_tokens
            )
            results.extend(results_batch)
            targets.extend(target_batch)
            total_token_num+= total_tokens_num_batch
            total_token_num_o1 += total_tokens_num_o1_batch

        # evaluate answers
        accuracy = eval_results(args.dataset, results, targets)
        logger.info("="*50 +f" Finished evaluation for {exp_id}-th experiment "+"="*50)
        logger.info({
            "Accuracy": f"{accuracy*100:.2f}%",
            "Dataset": args.dataset,
            "model": args.model,
            "batch size": args.bs,
            "problem nums": len(results),
            "total token num": total_token_num,
            "avg token num": total_token_num/len(results),
            "avg token num o1": total_token_num_o1/len(results),
            "exp": args.exp,
        })
        logger.info("="*50+f" Finish {exp_id}-th experiment "+"="*50)
        accuracy_across_exps.append(accuracy)
        token_num_across_exps.append(total_token_num)
        token_num_o1_across_exps.append(total_token_num_o1)
    # final results across repetitive experiments
    logger.info("="*50 + f" Finish {args.repeat_exp_num} exps " + "="*50)
    logger.info(f"Average accuracy over {args.repeat_exp_num} exps: {sum(accuracy_across_exps)/len(accuracy_across_exps)*100:.2f}%")
    logger.info(f"Average token num over {args.repeat_exp_num} exps: { sum(token_num_across_exps)/len(token_num_across_exps)/len(results) :.2f}")
    logger.info(f"Average token num o1 over {args.repeat_exp_num} exps: { sum(token_num_o1_across_exps)/len(token_num_o1_across_exps)/len(results) :.2f}")

    



