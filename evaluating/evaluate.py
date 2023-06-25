from transformers import pipeline, set_seed, AutoTokenizer, PegasusXForConditionalGeneration, DataCollatorForSeq2Seq
import matplotlib.pyplot as plt
from datasets import load_dataset, list_datasets, load_metric
import pandas as pd

import nltk
from nltk.tokenize import sent_tokenize

from tqdm import tqdm
import torch

from huggingface_hub import HfApi

import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print()
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    print()

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

print_gpu_utilization()
clear_gpu_memory()

nltk.download("punkt")

dataset = load_dataset("allenai/cord19", "fulltext")

print("[INFO] Loaded dataset")

filtered_dataset = dataset['train'].filter(lambda sample: sample['fulltext'] != '')
filtered_dataset = filtered_dataset.filter(lambda sample: sample['abstract'] != '')

train_dataset = filtered_dataset.train_test_split(test_size=0.1)['train']
validation_dataset = filtered_dataset.train_test_split(test_size=0.1)['test'].train_test_split(test_size=0.5, shuffle=False)['train']
test_dataset = filtered_dataset.train_test_split(test_size=0.1, shuffle=False)['test'].train_test_split(test_size=0.5, shuffle=False)['test']

print()
print("Training dataset size:", len(train_dataset))
print("Validation dataset size:", len(validation_dataset))
print("Test dataset size:", len(test_dataset))
print()


device = "cuda" if torch.cuda.is_available() else "cpu"

print()
print("[INFO] Loading model...")

# Change
model_ckpt = "aplnestrella/pegasus-x-cord19-ENC_16-DEC_16-b_8-e_8-g_1"

model = PegasusXForConditionalGeneration.from_pretrained(model_ckpt).to(device)
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-large")

print("[INFO] Loaded model")
print()

# print("[INFO] Printing preview...")
# for article in test_dataset:
#     print(article['doi'])

print()
print("[INFO] Calculating ROUGE scores...")
print()

def generate_summary(batch):
    inputs = tokenizer(batch['fulltext'], padding='max_length', truncation=True, max_length=16384, return_tensors='pt')
    summary_ids = model.generate(input_ids=inputs["input_ids"].to(device),
                         attention_mask=inputs["attention_mask"].to(device), 
                         length_penalty=0.8, num_beams=1, max_length=256)
    summaries = [tokenizer.decode(summary_id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for summary_id in summary_ids]
    return {'summary': summaries}

summaries = test_dataset.map(generate_summary, batched=True, batch_size=4)

rouge = load_metric('rouge')

rouge_output = rouge.compute(predictions=summaries['summary'], references=test_dataset['abstract'])

rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

print("Model:", model_ckpt)
print()

print("LOW")
rouge_dict = dict((rn, rouge_output[rn].low.fmeasure ) for rn in rouge_names )
print(pd.DataFrame(rouge_dict, index = ["Model"]))
print()

print()
print("MID")
rouge_dict = dict((rn, rouge_output[rn].mid.fmeasure ) for rn in rouge_names )
print(pd.DataFrame(rouge_dict, index = ["Model"]))
print()

print()
print("HIGH")
rouge_dict = dict((rn, rouge_output[rn].mid.fmeasure ) for rn in rouge_names )
print(pd.DataFrame(rouge_dict, index = ["Model"]))
print()

print(rouge_output)

print_gpu_utilization()