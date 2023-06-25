"""Script for fine-tuning Pegasus
Example usage:
  # use XSum dataset as example, with first 1000 docs as training data
  from datasets import load_dataset
  dataset = load_dataset("xsum")
  train_texts, train_labels = dataset['train']['document'][:1000], dataset['train']['summary'][:1000]
  
  # use Pegasus Large model as base for fine-tuning
  model_name = 'google/pegasus-large'
  train_dataset, _, _, tokenizer = prepare_data(model_name, train_texts, train_labels)
  trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset)
  trainer.train()
 
Reference:
  https://huggingface.co/transformers/master/custom_datasets.html
"""

# Fixed hyperparameters
batch_size = 8
epoch_number = 8
grad_acc_steps = 1

# Experimented hyperparameters
encoder_attention_heads = 16
decoder_attention_heads = 16

trained_model_name = "pegasus-x"+"-cord19"+"-ENC_"+str(encoder_attention_heads)+"-DEC_"+str(decoder_attention_heads)+"-b_"+str(batch_size)+"-e_"+str(epoch_number)+"-g_"+str(grad_acc_steps)

print("[INFO] Model name is: " + trained_model_name)
print()
print("[INFO] batch_size: " + str(batch_size))
print("[INFO] epoch_number: " + str(epoch_number))
print("[INFO] grad_acc_steps: " + str(grad_acc_steps))
print()
print("[INFO] encoder_attention_heads: " + str(encoder_attention_heads))
print("[INFO] decoder_attention_heads: " + str(decoder_attention_heads))
print()

from transformers import PegasusXForConditionalGeneration, AutoTokenizer, Trainer, TrainingArguments
import torch
import time
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from huggingface_hub import HfApi

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
    # del variables

def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        print(info.free)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")

class PegasusDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)

      
def prepare_data(model_name, 
                 train_texts, train_labels, 
                 val_texts=None, val_labels=None, 
                 test_texts=None, test_labels=None):
  """
  Prepare input data for model fine-tuning
  """
  tokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-large")

  prepare_val = False if val_texts is None or val_labels is None else True
  prepare_test = False if test_texts is None or test_labels is None else True

  def tokenize_data(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True)
    decodings = tokenizer(labels, truncation=True, padding=True)
    dataset_tokenized = PegasusDataset(encodings, decodings)
    return dataset_tokenized

  train_dataset = tokenize_data(train_texts, train_labels)
  val_dataset = tokenize_data(val_texts, val_labels) if prepare_val else None
  test_dataset = tokenize_data(test_texts, test_labels) if prepare_test else None

  return train_dataset, val_dataset, test_dataset, tokenizer


def prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset=None, freeze_encoder=False, output_dir='./results'):
  """
  Prepare configurations and base model for fine-tuning
  """
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(torch_device)
  from transformers import AutoConfig, AutoModel

  from transformers import PegasusXConfig, PegasusXModel

  # Load the PEGASUS configuration object
  config = PegasusXConfig.from_pretrained(model_name)

  # Edit the configuration as desired
  config.encoder_attention_heads = encoder_attention_heads
  config.decoder_attention_heads = decoder_attention_heads

  model = PegasusXForConditionalGeneration.from_pretrained(model_name, config=config).to(torch_device)

  if freeze_encoder:
    for param in model.model.encoder.parameters():
      param.requires_grad = False

  if val_dataset is not None:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      per_device_eval_batch_size=batch_size,    # batch size for evaluation, can increase if memory allows
      save_steps=500,                  # number of updates steps before checkpoint saves
      save_total_limit=2,              # limit the total amount of checkpoints and deletes the older checkpoints
      evaluation_strategy='steps',     # evaluation strategy to adopt during training
      eval_steps=10000,                  # number of update steps before evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      learning_rate=8e-4,

      # tweak to decrease training time
      num_train_epochs=epoch_number,           # total number of training epochs
      per_device_train_batch_size=batch_size,   # batch size per device during training, can increase if memory allows
      optim="adafactor",
      gradient_accumulation_steps=grad_acc_steps,
      logging_steps=10,
      fp16=False,
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=val_dataset,            # evaluation dataset
      tokenizer=tokenizer
    )

  else:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      save_steps=500,                  # number of updates steps before checkpoint saves
      save_total_limit=2,              # limit the total amount of checkpoints and deletes the older checkpoints
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      learning_rate=8e-4,

      # tweak to decrease training time
      num_train_epochs=epoch_number,           # total number of training epochs
      per_device_train_batch_size=batch_size,   # batch size per device during training, can increase if memory allows
      optim="adafactor",
      gradient_accumulation_steps=grad_acc_steps,
      logging_steps=10,
      fp16=False,
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      tokenizer=tokenizer
    )

  return trainer


if __name__=='__main__':
#   Use PEGASUS-X (base) model as for fine-tuning
  model_name = 'google/pegasus-x-base'

#   Use this for gauging available GPU memory
#   print("[INFO] Waiting for enough GPU memory...")
#   min_memory_available = 85028569088  # 2GB
#   clear_gpu_memory()
#   wait_until_enough_gpu_memory(min_memory_available)
#   print("[INFO] Enough GPU memory acquired")

  print_gpu_utilization()

  print("[INFO] Loading dataset...")
  from datasets import load_dataset
  dataset = load_dataset("allenai/cord19", "fulltext")
  print("[INFO] Loaded dataset")

  print_gpu_utilization()
  
  print("[INFO] Preparing dataset...")
  
  filtered_dataset = dataset['train'].filter(lambda sample: sample['fulltext'] != '')
  filtered_dataset = filtered_dataset.filter(lambda sample: sample['abstract'] != '')

  train_dataset = filtered_dataset.train_test_split(test_size=0.1, shuffle=False)['train']
  validation_dataset = filtered_dataset.train_test_split(test_size=0.1, shuffle=False)['test'].train_test_split(test_size=0.5, shuffle=False)['train']
  test_dataset = filtered_dataset.train_test_split(test_size=0.1, shuffle=False)['test'].train_test_split(test_size=0.5, shuffle=False)['test']

  train_texts, train_labels = train_dataset['fulltext'], train_dataset['abstract']
  validation_texts, validation_labels = validation_dataset['fulltext'], validation_dataset['abstract']
  train_dataset, validation_dataset, _, tokenizer = prepare_data(model_name, train_texts, train_labels, val_texts=validation_texts, val_labels=validation_labels)


  print()
  print("Training dataset size:", len(train_dataset))
  print("Validation dataset size:", len(validation_dataset))
  print("Test dataset size:", len(test_dataset))
  print()
  
  print("[INFO] Dataset prepared")

  print("[INFO] Loading model...")
  trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset=validation_dataset)
  print("[INFO] Loaded model")

  print_gpu_utilization()
  
  print("[INFO] Starting training...")
  result = trainer.train()

  print()
  print(result)
  print()

  print_gpu_utilization()
  print("[INFO] Finished training")