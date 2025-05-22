import json
import datasets 

def load_dataset(dataset_name):
    data = []
    if dataset_name=="gsm8k":
        data_main = datasets.load_dataset("openai/gsm8k", "main", split='test')
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

# dynamic injection position
def find_dynamic_punctuation(text, length):

    start_ratio = min( max(length-128, 0)/1024, 0.8)
    start =  int(start_ratio* (len(text)-1) )

    for i in range(start, len(text)-1 ):
        if text[i] in ('.', ','):
            if text[i+1]==' ':
                return i, text[i]
            
    return None, None

# create GPQA dataset
def create_and_save_gpqa():
    import datasets
    import random
    import json
    data_main = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond", split='train')
    data_list = []
    for data in data_main:
        options = [data['Correct Answer'].strip(), data['Incorrect Answer 1'].strip(), data['Incorrect Answer 2'].strip(), data['Incorrect Answer 3'].strip()]
        random.shuffle(options)
        print(options)
        item = {
            "question":f"{data['Question'].strip()}\nWhat is the correct answer to this question?\nA:{options[0]}\nB:{options[1]}\nC:{options[2]}\nD:{options[3]}\n",
            "answer": ['A','B','C','D'][options.index(data['Correct Answer'].strip())]
        }
        data_list.append(item)
        print(item)
    with open("gpqa-diamond.json", 'w', encoding='utf-8') as f:
      json.dump(data_list, f, indent=4)  # Use indent for readability
    