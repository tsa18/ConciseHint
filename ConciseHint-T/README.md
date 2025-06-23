### ConciseHint-T trains the hint embeddings on the concise dataset to further reduce the token usage.


## 🔧Setup

```bash
conda create -n concise_hint_t python=3.10 
conda activate concise_hint_t
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 🚀Run


### 1. Prepare training data (MixChain-Z-GSM8K)

```bash
python data_utils.py
```

### 2. Finetune hint embeddings on MixChain-Z-GSM8K

```bash
CUDA_VISIBLE_DEVICES=0 python finetuning_concise.py
```



### 3. Run inference

```bash
CUDA_VISIBLE_DEVICES=0 python -u inference_concise.py --model  output/Qwen3_1.7B-finetuning_concise_level-0/checkpoint-7000  --dataset gsm8k   --max-tokens 10240  --exp qwen-3-1.7b-train_gsm8k-test-gsm8k_ours  --bs 64       
```
