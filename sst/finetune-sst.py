## Fine-Tuning - Sentiment Analysis 

## Import Libraries
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset

## Import Torch Libraires
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

## Other libraries
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
      padding = 'max_length',
      return_attention_mask=True,
      return_tensors='pt')

    return {
      'sentences': sentence,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'labels': torch.tensor(label, dtype=torch.long)
    }
  
## Function for training the model
def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples, scheduler):
  model = model.train()

  losses = []
  correct_predictions = 0

  for d in tqdm(data_loader):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["labels"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()
    scheduler.step()
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

      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

## Function to train 
def train(train_data_loader, val_data_loader, learning_rate, weight_decay, device, n_epochs, df_train_len, df_val_len, save_path, pretrained_model_path, model_name):
  model = SentimentClassifier(2, pretrained_model_path)
  model = model.to(device)
  
  history = defaultdict(list)

  optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, correct_bias=False)

  total_steps = len(train_data_loader) * n_epochs
  warmup_steps = int(total_steps * 0.1)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

  loss_fn = nn.CrossEntropyLoss().to(device)

  # Initialize wandb
  wandb.init(project="sst-final-v5", entity="imp-language", name = model_name)

  for epoch in range(n_epochs):
    
    print(f'Epoch {epoch + 1}/{n_epochs}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, df_train_len, scheduler)

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, df_val_len)
    print(f'Val loss {val_loss} accuracy {val_acc}')

    history['train_acc'].append(train_acc.cpu().numpy().item())
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc.cpu().numpy().item())
    history['val_loss'].append(val_loss)

    wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

    saving_path = f'{save_path}/epoch_{epoch+1}'
    os.makedirs(saving_path, exist_ok=True)
    save_model_and_training_state(model, tokenizer, optimizer, scheduler, saving_path)
    
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

  return DataLoader(ds, batch_size=batch_size, shuffle=True)

def save_model_and_training_state(model, tokenizer, optimizer, scheduler, save_path):
    os.makedirs(save_path, exist_ok=True)

    # Save the model weights
    model_save_path = os.path.join(save_path, 'pytorch_model.bin')
    torch.save(model.state_dict(), model_save_path)

    ## Save another with .pt
    model2_save_path = os.path.join(save_path, 'model.pt')
    torch.save(model, model2_save_path)

    # Save the optimizer state
    optimizer_save_path = os.path.join(save_path, 'optimizer.pt')
    torch.save(optimizer.state_dict(), optimizer_save_path)

    # Save the scheduler state
    scheduler_save_path = os.path.join(save_path, 'scheduler.pt')
    torch.save(scheduler.state_dict(), scheduler_save_path)

    # Save the tokenizer
    tokenizer.save_pretrained(save_path)

## Define model
class SentimentClassifier(nn.Module):

    def __init__(self, n_classes, pretrained_model_path):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)

        # Define MLP classification head
        # self.classification_head = nn.Sequential(
          # nn.Dropout(0.1),
          # nn.Linear(768, 2))
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes))

        # Initialize MLP layers with Kaiming (He) initialization
        for m in self.classification_head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        return self.classification_head(pooled_output)

if __name__ == '__main__':

    ## Hyperparameters
    EPOCHS = 5
    
    TRAIN_BATCH_SIZE = 64
    VAL_BATCH_SIZE = 64
    MAX_LEN = 80

    ## Set DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Set seeds
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    ## Relevant paths 
    DATA_PATH = "/home/kaydin/sst"
    MODEL_LOAD = "/netscratch2/kaydin/mlm-nsp-bert/pretraining"
    MODEL_SAVE = "/netscratch2/kaydin/sst-2"   

    ## Load data
    df_train = pd.read_json(f'{DATA_PATH}/train.json', lines=True)
    df_val = pd.read_json(f'{DATA_PATH}/val.json', lines=True)
    
    ## Hyperparameters Range
    LEARNING_RATES = [
      # 0.00005, 
      0.00002,
      0.00003,
      ]
    
    WEIGHT_DECAYS = [
      0.01, 
      # 0.0001,
      0.001
    ]

    ## Chosen models
    CHOSEN_MODELS = [
      #"continued_lr_0.00005_wd_0.0001", 
      #"continued_lr_0.00005_wd_0.01", 
      #"continued_lr_0.00003_wd_0.0001", 
      #"continued_lr_0.00003_wd_0.01", 
      #"continued_lr_0.0001_wd_0.0001", 
      #"continued_lr_0.0001_wd_0.01", 
      #"scratch_lr_0.0001_wd_0.0001",
      #"scratch_lr_0.0001_wd_0.01",
      "seed_lr_0.0001_wd_0.01_seed_1",
      "seed_lr_0.0001_wd_0.01_seed_2",
      # "bert-base-cased"
      ]  

    for CHOSEN_MODEL in CHOSEN_MODELS:

      ## Path to pre-trained model
      # MODEL_LOAD_PATH = os.path.join(MODEL_LOAD, CHOSEN_MODEL, "final")
      MODEL_LOAD_PATH = os.path.join(MODEL_LOAD, CHOSEN_MODEL)

      ## Load tokenizer  
      tokenizer = BertTokenizer.from_pretrained(MODEL_LOAD_PATH)

      ## Iterate through hyperparameters
      for LEARNING_RATE in LEARNING_RATES:
        for WEIGHT_DECAY in WEIGHT_DECAYS:

          print("PT Model:", CHOSEN_MODEL)
          print("Learning Rate:", LEARNING_RATE)
          print("Weight Decay:", WEIGHT_DECAY)

          ## Name & Path
          MODEL_NAME = f'{CHOSEN_MODEL}_sst_lr_{LEARNING_RATE}_wd_{WEIGHT_DECAY}'
          MODEL_SAVE_PATH = os.path.join(MODEL_SAVE, MODEL_NAME)
          os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

          ## Load dataloader
          train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, TRAIN_BATCH_SIZE)
          val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, VAL_BATCH_SIZE)

          ## Start training
          train(train_data_loader, val_data_loader, LEARNING_RATE, WEIGHT_DECAY, DEVICE, EPOCHS, len(df_train), len(df_val), MODEL_SAVE_PATH, MODEL_LOAD_PATH, MODEL_NAME)