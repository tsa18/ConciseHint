
import json
import datasets

def prepare_mixchain_gsm8k():
    data_list = datasets.load_dataset("horseee/MixChain-Z-GSM8K", split='train')
    items = []
    with open('MixChain-Z-GSM8K_level-0.jsonl', 'w') as f:
        for data in data_list:
            for k in range(0,5):
                if k!=0: continue
                item = {
                    'question': data['question'],
                    'target':  str(data['target']),
                    'answer': data['answer'],
                    f'solution_{k}': data[f'solution_{k}'],
                    'length_level': k
                }
                
                item_format = {
                    "length_level": k,
                    "input": item['question'].strip(),
                    "output": "<think>"+'\n' + item[f'solution_{k}'].strip() + '\n' + "</think>" + "\n\n" + item["answer"].strip(),
                }
                f.write(json.dumps(item_format) + '\n')
        # print(item_format)



# load dataset for inference
def load_dataset(dataset_name):
    data = []
    if dataset_name=="gsm8k":
        data_main = datasets.load_dataset("openai/gsm8k", "main", split='test')
        for d in data_main:
            data.append({
                'problem': d['question'].strip(),
                'answer': f"{d['answer']}",
            })
    elif dataset_name=="gsm8k_dev":
        data_main = datasets.load_dataset("openai/gsm8k", "main", split='train')
        for d in data_main:
            data.append({
                'problem': d['question'].strip(),
                'answer': f"{d['answer']}",
            })
    elif dataset_name=="aime24":
        data_main = datasets.load_dataset("Maxwell-Jia/AIME_2024", split="train")
        for d in data_main:
            data.append({
                'problem': d['Problem'].strip(),
                'answer': f"{d['Answer']}".strip(),
            })
    elif dataset_name=="gpqa_diamond":
        with open('./data/gpqa-diamond.json', 'r', encoding='utf-8') as file:
            data_main = json.load(file)
        for d in data_main:
            data.append({
                'problem': d['question'].strip(),
                'answer': f"{d['answer']}".strip(),
            })
    else:
        raise NotImplementedError
    return data

    

if __name__ == "__main__":
    prepare_mixchain_gsm8k()

