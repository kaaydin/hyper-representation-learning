## Fine-Tuning - Sentiment Analysis 

## Import Libraries
from transformers import BertModel, BertTokenizer, AdamW
from datasets import load_dataset

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from collections import defaultdict
import json
from tqdm import tqdm
import wandb
import os

## Define Dataset
class SSTDataset(Dataset):

  def __init__(self, sentences, labels, tokenizer, max_len):
    self.sentences = sentences
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.sentences)

  def __getitem__(self, item):
    sentence = str(self.sentences[item])
    label = self.labels[item]

    encoding = self.tokenizer.encode_plus(
      sentence,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'sentences': sentence,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'labels': torch.tensor(label, dtype=torch.long)
    }
  
## Function for training the model
def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0

  for d in tqdm(data_loader):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["labels"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

## Function for evaluating the model
def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in tqdm(data_loader):
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["labels"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

## Function to train 
def train(train_data_loader, val_data_loader, learning_rate, device, n_epochs, df_train_len, df_val_len, save_path, pretrained_model_path, model_name):
  model = SentimentClassifier(2, pretrained_model_path)
  model = model.to(device)

  os.makedirs(save_path, exist_ok=True)
  
  history = defaultdict(list)

  optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
  loss_fn = nn.CrossEntropyLoss().to(device)

  # Initialize wandb
  wandb.init(project="SST Finetuning", 
              entity="imp-language",
              name = model_name,
              config={
      "learning_rate": learning_rate,
      "max_epochs": n_epochs
  })

  for epoch in range(n_epochs):
    
    print(f'Epoch {epoch + 1}/{n_epochs}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
      model,
      train_data_loader,
      loss_fn,
      optimizer,
      device,
      df_train_len
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
      model,
      val_data_loader,
      loss_fn,
      device,
      df_val_len
    )

    print(f'Val loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc.cpu().numpy().item())
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc.cpu().numpy().item())
    history['val_loss'].append(val_loss)

    wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

    saving_path = f'{save_path}/epoch_{epoch+1}'
    os.makedirs(saving_path)
    torch.save(model.state_dict(), f'{saving_path}/model_state.bin')
    
  ## Close wandb
  wandb.finish()

  ## Save history
  with open(f'{save_path}/history.json', 'w') as outfile:
    json.dump(history, outfile)

# Define Data Loader
def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = SSTDataset(sentences=df.sentence.to_numpy(),
                        labels=df.label.to_numpy(),
                        tokenizer=tokenizer,
                        max_len=max_len)

  return DataLoader(ds, batch_size=batch_size)


## Define model
class SentimentClassifier(nn.Module):

  def __init__(self, n_classes, pretrained_model_path):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(pretrained_model_path)
    self.drop = nn.Dropout(p=0.1)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    a = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    pooled_output = a.pooler_output
    output = self.drop(pooled_output)
    return self.out(output)


if __name__ == '__main__':

    ## Hyperparameters
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    TRAIN_BATCH_SIZE = 64
    VAL_BATCH_SIZE = 64
    MAX_LEN = 80

    ## Set DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Set seeds
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    ## Relevant paths 
    DATA_PATH = "/Users/kaanaydin/Library/CloudStorage/GoogleDrive-implanguagetransformers@gmail.com/My Drive/imp-project/downstream_tasks/sst/data"
    # DATA_PATH = "/home/kaydin/data/sst"

    MODEL_LOAD = "/Users/kaanaydin/Library/CloudStorage/GoogleDrive-implanguagetransformers@gmail.com/My Drive/imp-project/pretraining/models"
    #MODEL_LOAD = "/netscratch2/kaydin/models"
    
    MODEL_SAVE = "/Users/kaanaydin/Library/CloudStorage/GoogleDrive-implanguagetransformers@gmail.com/My Drive/imp-project/downstream_tasks/sst/models"
    # MODEL_SAVE = "/netscratch2/kaydin/sst"

    ## Load data
    df_train = pd.read_json(f'{DATA_PATH}/train.json', lines=True)
    df_val = pd.read_json(f'{DATA_PATH}/val.json', lines=True)

    print("Data is loaded")

    ## Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    ## Load Dataset
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, TRAIN_BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, VAL_BATCH_SIZE)

    ## Model List 
    CHOSEN_MODEL_LIST = ["bert_lr_0.0001_wd_0.01", "bert_lr_0.0001_wd_0.0001"]
    CHOSEN_EPOCH_LIST = [10, 5, 1]

    ## Train on BERT standard
    # MODEL_NAME_BERT = "bert-base-cased"
    # MODEL_NAME_FINETUNED = f'{MODEL_NAME_BERT}_sst_lr_{LEARNING_RATE}'
    # MODEL_SAVE_PATH_BERT = f'{MODEL_SAVE}/{MODEL_NAME_FINETUNED}'
    # print(MODEL_NAME_BERT)
    # train(train_data_loader, val_data_loader, LEARNING_RATE, DEVICE, EPOCHS, len(df_train), len(df_val), MODEL_SAVE_PATH_BERT, MODEL_NAME_BERT, MODEL_NAME_BERT)

    ## Train on BERT with different learning rates & models
    for CHOSEN_MODEL in CHOSEN_MODEL_LIST:
      for CHOSEN_EPOCH in CHOSEN_EPOCH_LIST:
        MODEL_NAME = f'{CHOSEN_MODEL}_epoch{CHOSEN_EPOCH}_sst_lr_{LEARNING_RATE}'
        MODEL_LOAD_PATH = f'{MODEL_LOAD}/{CHOSEN_MODEL}/epoch_{CHOSEN_EPOCH}'
        MODEL_SAVE_PATH = f'{MODEL_SAVE}/{MODEL_NAME}'
        print(MODEL_NAME)
        train(train_data_loader, val_data_loader, LEARNING_RATE, DEVICE, EPOCHS, len(df_train), len(df_val), MODEL_SAVE_PATH, MODEL_LOAD_PATH, MODEL_NAME)