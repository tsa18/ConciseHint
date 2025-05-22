import argparse
from tqdm import tqdm
from utils import load_dataset
from clients import vllmClientModel
from tqdm import trange
from pytorch_lightning import seed_everything
seed_everything(999)
from logger import setup_logger
import logging

import tiktoken
# set tokenizer
tokenizer = tiktoken.encoding_for_model("o1")

# measure tge token usage 
def token_measure(text):
    """
    Count the number of tokens in a text using o1-like tokenizer.
    
    Args:
        text: Text to measure

    Returns:
        int: Number of tokens in the text
    """
    return len(tokenizer.encode(text, disallowed_special=()))


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
        "--url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL listened by the vLLM server",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="token-abc123",
        help="API key of the model to use",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start index of the question"
    )
    parser.add_argument(
        "--end", type=int, default=100000000, help="End index of the question"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum number of tokens per request",
    )
    parser.add_argument(
        "--port", type=int, default=8000
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument(
        "--bs", type=int, default=16, help="batch size to process questions"
    )
    parser.add_argument("--top-p", type=float, default=0.95, help="Top p for sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exp", type=str, default="exp", help="annotation for this exp")
    parser.add_argument("--repeat_exp_num", default=5, type=int, help="The number of repetitive exps")
    parser.add_argument("--enable_adap", action='store_true', help="if enable AdaP prompt")
    parser.add_argument("--enable_hint", action='store_true', help="if enable ConciseHint")

    return parser.parse_args()


def load_model(model_name, url, api_key):
    return vllmClientModel(model_name, url, api_key)


print_sample = True

# parallelly generate answers for a batch of questions
def execute_questions_batch(
    args,
    model,
    questions_batch,
    top_p=0.95,
    temperature=0.6,
    max_token_length = 16384,
):

    # textual hint
    hint_prompt = "-<|im_start|>user\nmake answer concise!<|im_end|>-"  
    # original input prompts
    input_prompts = [model.prepare_prompt(question) for question in questions_batch]
    # enable AdaP/hint, if enable_adap is true, AdaP is activated. If enable_hint is true, hint injection is activated.
    enable_adap = args.enable_adap
    enable_hint = args.enable_hint

    total_tokens_num = 0


    ### modifiy the input prompts
    if args.dataset=="gpqa_diamond":
        if 'Qwen3' in args.model:
            if enable_adap:
                input_prompts =  ["<|im_start|>user\n" + prompt.replace("<|User|>","").replace("<|Assistant|>", "\nGive your final answer in the format: The correct answer is <Your choice among A, B, C, and D>. Please adaptively control the answer length based on the query's complexity. The lower the complexity, the more concise your answer should be.<|im_end|>\n<|im_start|>assistant\n") for prompt in input_prompts] 
            else:
                input_prompts =  ["<|im_start|>user\n" + prompt.replace("<|User|>","").replace("<|Assistant|>", "\nGive your final answer in the format: The correct answer is <Your choice among A, B, C, and D>.<|im_end|>\n<|im_start|>assistant\n") for prompt in input_prompts] 
        else:
            if enable_adap:
                input_prompts = [prompt.replace("<|Assistant|>", "\nGive your final answer in the format: The correct answer is <Your choice among A, B, C, and D>. Please adaptively control the answer length based on the query's complexity. The lower the complexity, the more concise your answer should be.\n"+"<|Assistant|>") for prompt in input_prompts] 
            else:
                input_prompts = [prompt.replace("<|Assistant|>", "\nGive your final answer in the format: The correct answer is <Your choice among A, B, C, and D>.\n"+"<|Assistant|>") for prompt in input_prompts] 
    elif args.dataset=="gsm8k" or args.dataset=="aime24":
        if 'Qwen3' in args.model:
            if enable_adap:
                input_prompts  = ["<|im_start|>user\n"+prompt.replace("<|User|>","").replace("<|Assistant|>", "\nPut your final answer within \\boxed{}. Please adaptively control the answer length based on the query's complexity. The lower the complexity, the more concise your answer should be.<|im_end|>\n<|im_start|>assistant\n") for prompt in  input_prompts] 
            else:
                input_prompts  = ["<|im_start|>user\n"+prompt.replace("<|User|>","").replace("<|Assistant|>", "\nPut your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n") for prompt in  input_prompts]
        else:
            if enable_adap:
                input_prompts = [prompt.replace("<|Assistant|>", "\nPut your final answer within \\boxed{}. Please adaptively control the answer length based on the query's complexity. The lower the complexity, the more concise your answer should be.\n"+"<|Assistant|>") for prompt in input_prompts] 
            else:
                input_prompts  = [prompt.replace("<|Assistant|>", "\nPut your final answer within \\boxed{}.\n"+"<|Assistant|>") for prompt in  input_prompts]
    else:
        raise NotImplementedError


    # if the question is finished, stopped or exceed the max length
    is_finished = [False] * len(input_prompts)
    # record completions of current round of generation
    completions = [""] * len(input_prompts)
    # record entire generation
    results = [""] * len(input_prompts)
    # alpha in the paper
    base_len = 128
    current_lens = [0] * len(input_prompts)
    # injection interval
    interval_token_num = [base_len] * len(input_prompts)
    pbar = tqdm(total=len(input_prompts), desc="Processing single question", unit="step")
    current_prompts = input_prompts[:]

    while True:
        if all(is_finished):
            break
        # use fixed interval if hint is not enabled
        if (not enable_hint): 
            interval_token_num = [base_len] * len(input_prompts)
        # generate...
        responses = model.generate_batch(
                current_prompts,
                max_tokens= interval_token_num, 
                is_actives= [not finished for finished in is_finished],
                top_p=top_p,
                temperature=temperature,
            )
        # post-process
        for i in range(len(current_prompts)):

            if is_finished[i]: continue

            response = responses[i]
            text = response.choices[0].text
            finish_reason = response.choices[0].finish_reason
            completions[i] = text

            # if reach the final answer rather than exceed the length limitation
            if finish_reason != "length": 
                    is_finished[i] = True
                    overall_output = current_prompts[i] + completions[i]
                    # print('='*50+f' Completed {i}-th question in batch'+'='*50)
                    token_num = token_measure(overall_output)
                    total_tokens_num +=token_num
                    global print_sample
                    if print_sample:
                        print('-'*100)
                        print(overall_output) # debug
                        print('token num:',token_num )
                        print('-'*100)
                        print_sample = False
                        print('='*50+' End '+'='*50)
                    results[i] =  overall_output.replace(hint_prompt,"").replace(input_prompts[i],"") 
                    pbar.update(1)

            if enable_hint:
                from utils import find_dynamic_punctuation
                # dynamically find the injection position
                pos, _ = find_dynamic_punctuation(completions[i], length=interval_token_num[i])
                if pos is not None:
                    current_prompts[i] = (current_prompts[i] + completions[i][0:pos+1])+ f"{hint_prompt}" +  completions[i][pos+1:] # inject hint
                else:
                    current_prompts[i] = (current_prompts[i] + completions[i])# ori
            else:
                current_prompts[i] =  (current_prompts[i] + completions[i]) # ori

            current_lens[i]+=interval_token_num[i]
            # update the injection interval
            interval_token_num[i] = min(1024, int( base_len + 0.2*token_measure(current_prompts[i]))) 

            # terminate if exceed the max length
            if token_measure(current_prompts[i]) > max_token_length: 
                is_finished[i] = True
                overall_output = current_prompts[i]
                token_num = token_measure(overall_output)
                total_tokens_num +=token_num
                results[i] =  overall_output.replace(hint_prompt,"").replace(input_prompts[i],"")
                pbar.update(1)
                print(f"Token length={token_measure(current_prompts[i])}, Exceed max len, truncate.")
     
    logger = logging.getLogger(name="my_logger")
    logger.info(f"total-tokens in batch: {total_tokens_num}",  )
    logger.info('-'*100)
    return results, total_tokens_num 


from evaluator import Evaluator
from grader import grade_answer
# eval if the answer is correct
def eval_results(dataset, results, targets):
    correct_num = 0
    for result, target in zip(results, targets):
        if dataset == "gsm8k":
            target = Evaluator.find_answer_gsm8k(target)
        if dataset=="gsm8k" or dataset=="aime24":
            correct = (target==Evaluator.extract_predicted_answer(result)) or (r"\boxed{"+str(target)+r"}" in result) \
                    or grade_answer(target,  Evaluator.extract_predicted_answer(result))
            extracted_last_k = Evaluator.extract_predicted_answer(result, last=3) 
            correct = correct or (isinstance(extracted_last_k, tuple) and any(grade_answer(target, answer) for answer in extracted_last_k)) # corner cases where \boxed is not in answer and the last number is also not the answer
        elif dataset=="gpqa_diamond":
            correct = f"The correct answer is {target}" in result or f"The correct answer is **{target}**" in result or f"The correct answer is **{target}" in result \
                       or f"The correct answer is <{target}>" in result  or f"The correct answer is <{target}" in result or r"\boxed{"+str(target) in result \
                       or r"\boxed{\text{"+str(target) in result
        else:
            raise NotImplementedError
        correct_num += correct
            
    accuracy = correct_num/len(results)
    return accuracy


def main():

    args = parse_args()
    data = load_dataset(args.dataset)
    args.url = f"http://localhost:{args.port}/v1"

    model = load_model(args.model, args.url, args.api_key)
    bs = args.bs
    logger = setup_logger(name="my_logger", log_dir='experiment_logs', log_filename=f"{args.model.split('/')[-1]}-{args.dataset}-{args.repeat_exp_num}-bs={args.bs}-exp-{args.exp}.log") 

    accuracy_across_exps = []
    token_num_across_exps = []
    for exp_id in range(args.repeat_exp_num):
        logger.info("="*50+f" Start {exp_id}-th experiment "+"="*50)
        results, targets = [], []
        total_token_num = 0
        for s in trange(0, len(data), bs, desc="Processing questions batch"):
            if s < args.start: continue
            elif s > args.end: break
            data_batch = data[s: s+bs]
            questions_batch = [ item["problem"].strip() for item in data_batch]
            target_batch = [ item["answer"].strip() for item in data_batch]
            # generate answers for a batch of questions
            results_batch, total_tokens_num_batch  = execute_questions_batch(
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
            "exp": args.exp,
        })
        logger.info("="*50+f" Finish {exp_id}-th experiment "+"="*50)
        accuracy_across_exps.append(accuracy)
        token_num_across_exps.append(total_token_num)
    # final results across repetitive experiments
    logger.info("="*50 + f" Finish {args.repeat_exp_num} exps " + "="*50)
    logger.info(f"Average accuracy over {args.repeat_exp_num} exps: {sum(accuracy_across_exps)/len(accuracy_across_exps)*100:.2f}%")
    logger.info(f"Average token num over {args.repeat_exp_num} exps: { sum(token_num_across_exps)/len(token_num_across_exps)/len(results) :.2f}")


if __name__ == "__main__": 
    main()
