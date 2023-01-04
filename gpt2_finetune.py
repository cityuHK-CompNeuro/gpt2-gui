# GPT-2 Fine-Tuning with PyTorch, Huggingface, Amazon Reviews Dataset

# This is a simplified script for fine-tuning GPT2 using Hugging Face's [Transformers library](https://huggingface.co/transformers/) and PyTorch.
# 
# This is notebook is mostly from: https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh  
# 
# The original author cites these tutorials:  
# [Chris McCormick's BERT fine-tuning tutorial](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)  
# [Ian Porter's GPT2 tutorial](https://snappishproductions.com/blog/2020/03/01/chapter-9.5-text-generation-with-gpt-2-and-only-pytorch.html.html)  
# [Hugging Face Language model fine-tuning script](https://huggingface.co/transformers/v2.0.0/examples.html#language-model-fine-tuning) so full credit to them.  
# Chris' code has pretty much provided the basis for this script - you should definitely check out his [blog](https://mccormickml.com/tutorials/).
# 
# Dataset can be found here:  
# [Amazon Reviews 2018 dataset](https://jmcauley.ucsd.edu/data/amazon/) 

# %%
import os
import time
import datetime
import json
import math
import torch
import evaluate
import tqdm

import os.path as osp
import pandas as pd
import seaborn as sns
import numpy as np
import random
import pickle

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from tqdm import tqdm
from datetime import datetime, timedelta


torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import TrainingArguments, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
from torch import nn
from transformers import DataCollatorForLanguageModeling

# %%
output_dir = './model_save/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# %%

data_dir = 'D:\Coding\sandbox\Amzn_Stats\Amzn_data'
osp.isdir(data_dir)

dat_files = ['CDs_and_Vinyl_5', 'Prime_Pantry_5', 'Pet_Supplies_5']

# Read Reviews File

with open(osp.join(data_dir, dat_files[0] + '.json'),'r') as f:
    texts = f.readlines()

full_review = [json.loads(txt) for txt in texts]
print(full_review[0])

review_text = []
for fr in tqdm(full_review):
    try:
        review_text.append(fr['reviewText'])
    except:
        pass
print(f'Extracted {len(review_text)} text reviews')
print('First 5 reviews:')

for i, rt in enumerate(review_text[:5]):
    print(f'Review # {i}')
    print(rt)
    print('----------------------------- ')

# %%
recommended_text = []
for fr in tqdm(full_review):
    if 'vote' in list(fr.keys()):
        # print(fr['vote'])
        try:
            recommended_text.append(fr['reviewText'])
        except:
            pass
print(f'Extracted {len(recommended_text)} text reviews')
print('First 5 reviews:')

for i, rt in enumerate(recommended_text[:5]):
    print(f'Review # {i}')
    print(rt)
    print('----------------------------- ')

# %% [markdown]
# Create Training Set

# GPT2 is a large model. Increasing the batch size above 2 has lead to out of memory problems. This can be mitigated by accumulating the gradients but that is out of scope here.

# I'm using the standard PyTorch approach of loading data in using a [dataset class](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).
# 
# I'm passing in the tokenizer as an argument but normally I would  instantiate it within the class.

# %%
class GPT2Dataset(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="distilgpt2", max_length=512):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for txt in txt_list:

      # encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
      encodings_dict = tokenizer(txt, truncation=True, max_length=max_length, padding="max_length")


      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx] 

# %%
# To Load Previously Fine-Tuned Model
# Load a trained model and vocabulary that you have fine-tuned
load_dir = 'D:\\Coding\\sandbox\\gpt2_train\\model_save\\all3'
tokenizer = GPT2Tokenizer.from_pretrained(load_dir)
model = GPT2LMHeadModel.from_pretrained(load_dir)

tokenizer.pad_token = tokenizer.eos_token
dataset = GPT2Dataset(recommended_text, tokenizer, max_length=512)

# Split into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))



# %%
batch_size = 6

print(f'Batch size: {batch_size}')

# %%
# Create the DataLoaders for our training and validation datasets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

# # Finetune GPT2 Language Model

# %%
# # instantiate the model (if not loading from fine-tuned above)
# model = GPT2LMHeadModel.from_pretrained("distilgpt2", config=configuration)

device = torch.device("cuda")
model.to(device)

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Training Hyperparameters
epochs = 250
learning_rate = 5e-6
warmup_steps = 1e2
epsilon = 1e-8

# this produces a text generation sample output every n steps
sample_every = 100000

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )

# %%
# Total number of training steps is [number of batches] x [number of epochs]. 
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)

# %%
def format_time(elapsed):
    return str(timedelta(seconds=int(round((elapsed)))))

# %%
total_t0 = time.time()

training_stats = []

now = datetime.now()
run_ID = now.strftime("%Y%m%d_%H%M")

# Create model directory if needed
model_save_dir = osp.join(output_dir, run_ID)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

model_trainsave_dir = osp.join(output_dir, run_ID + '_train')
if not os.path.exists(model_trainsave_dir):
    os.makedirs(model_trainsave_dir)

best_val_loss = 100

# Run Training Loop
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print(f'Beginning Run {run_ID}')
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        model.zero_grad()        

        outputs = model(  b_input_ids,
                          labels=b_labels, 
                          attention_mask = b_masks,
                          token_type_ids=None
                        )

        loss = outputs[0]  

        batch_loss = loss.item()
        total_train_loss += batch_loss

        # Get sample every x batches.
        if step % sample_every == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

            model.eval()

            sample_outputs = model.generate(
                                    bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=50, 
                                    max_length = 200,
                                    top_p=0.95, 
                                    num_return_sequences=1
                                )
            for i, sample_output in enumerate(sample_outputs):
                print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

            model.save_pretrained(model_trainsave_dir)
            tokenizer.save_pretrained(model_trainsave_dir)

            
            model.train()

        loss.backward()

        optimizer.step()

        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)       
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        
        with torch.no_grad():        

            outputs  = model(b_input_ids, 
#                            token_type_ids=None, 
                             attention_mask = b_masks,
                            labels=b_labels)
          
            loss = outputs[0]  
            
        batch_loss = loss.item()
        total_eval_loss += batch_loss        

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    validation_time = format_time(time.time() - t0)    

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

    if avg_val_loss < best_val_loss:
        model.save_pretrained(model_save_dir)
        tokenizer.save_pretrained(model_save_dir)
        stats_file = osp.join(model_save_dir, 'train_stats')
        with open(stats_file, 'ab') as f:
            # source, destination
            pickle.dump(training_stats, f)                     

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

# %% [markdown]
# Summary of the training process.

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])

image_name = osp.join(model_save_dir, "Training_Loss.png")
plt.savefig(image_name)

plt.show()

# %% [markdown]
# # Display Model Info

# %%
# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The GPT-2 model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:2]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[2:14]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-2:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# %% [markdown]
# # Saving & Loading Fine-Tuned Model
# 

# %%
print(f"Saving model to {model_save_dir}")

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # remove distributed/parallel training if necessary
model_to_save.save_pretrained(model_save_dir)
tokenizer.save_pretrained(model_save_dir)

# %%
# Load a fine-tuned model
model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
model.to(device)

# %% [markdown]
# # Generate Text

# %%
model.eval()

prompt = "cindy manilow was"

generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
generated = generated.to(device)

print(generated)

sample_outputs = model.generate(generated, 
                                #bos_token_id=random.randint(1,30000),
                                do_sample=True,   
                                top_k=10, 
                                max_length = 50,
                                top_p=0.97, 
                                num_return_sequences=5
                                )

for i, sample_output in enumerate(sample_outputs):
  print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))


