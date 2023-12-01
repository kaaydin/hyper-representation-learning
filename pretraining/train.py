## Title: Pre-training for BERT

## Import libraries
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling
from tqdm import tqdm
import random
import pickle
import torch
from torch import nn
import time 
import os
import pandas as pd
import wandb

## Function to initialize weights
def init_weights(module):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()

## Function to initialize model
def model_init(MODEL_CHECKPOINT, device, model_resize = False, reinit = False):
    model = BertForMaskedLM.from_pretrained(MODEL_CHECKPOINT)

    # Adjusting the positional embeddings
    # The original max position embeddings is 512, we're changing it to 256
    if model_resize == True:
    
        new_max_position_embeddings = 256
        old_max_position_embeddings = model.config.max_position_embeddings

        # Resize the positional embeddings
        model.resize_token_embeddings(new_max_position_embeddings)

        # Truncate the positional embeddings if the new length is shorter
        if new_max_position_embeddings < old_max_position_embeddings:
            model.bert.embeddings.position_embeddings.weight.data = model.bert.embeddings.position_embeddings.weight.data[:new_max_position_embeddings, :]

        # Update the configuration
        model.config.max_position_embeddings = new_max_position_embeddings

    # Reinitialize the model
    if reinit == True:
        model.apply(init_weights)

    model.to(device)
    
    return model

## Functin to calcualte mlm scores
def calculate_mlm_scores(outputs, labels):
        
        ## ...
        logits = outputs.logits
        
        ## ...
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        ## ...
        predictions = torch.argmax(probabilities, dim=-1)
        
        ## ...
        mask = (labels != -100)
        
        ## ...
        correct = (predictions == labels) * mask

        ## ...
        correct = correct.sum().item()
        total = mask.sum().item()

        return correct, total

# %%
def train(model_checkpoint, device, train_dataloader, val_dataloader, max_epochs, learning_rate, weight_decay):

    ## Set model name
    MODEL_NAME = f"bert_lr_{learning_rate}_wd_{weight_decay}"

    ## Set model path 
    MODEL_SAVE_PATH = f"{MODEL_PATH}/{MODEL_NAME}"

    ## Create output directory
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    ## Set wandb notebook name
    os.environ["WANDB_NOTEBOOK_NAME"] = "pretraining.ipynb"

    # Initialize wandb
    wandb.init(project="pre-training-v3", 
               entity="imp-language",
               name = MODEL_NAME,
               config={
        "model_checkpoint": model_checkpoint,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs
    })
    
    ## Initialize model
    model = model_init(MODEL_CHECKPOINT, DEVICE, model_resize = False, reinit = False)

    ## Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    ## Tracker for losses and accuracy
    tracker = {
        "epochs": [],
        "train_steps": [],
        "val_steps": [],
        "train_loss_tracker": [],
        "train_accuracy_tracker": [],
        "val_loss_tracker": [],
        "val_accuracy_tracker": [],
        "train_time": []
    }

    ## ...
    step_train = 0
    step_val = 0

    for epoch in range(max_epochs):
        
        ## Set up starting time -> move to below
        START_TIME = time.time()

        print(f"Epoch: {epoch+1}/{max_epochs}")
        
        ## Append epoch to tracker
        tracker["epochs"].append(epoch+1)
        
        ## Initialize accumulators for each epoch

        total_train_loss = 0
        total_train_accuracy_correct = 0
        total_train_accuracy_total = 0
        total_val_accuracy_correct = 0
        total_val_accuracy_total = 0
        total_val_loss = 0
        total_val_accuracy = 0

        ## Set model to train mode
        model.train()

        for batch in tqdm(train_dataloader):
            
            ## ...
            optimizer.zero_grad()
            
            ## ...
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            ## ...
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            ## 
            
            ## ...
            correct, total = calculate_mlm_scores(outputs, labels)
            
            ## ...
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            ## Tracker for training loss
            total_train_loss += loss.item()
            total_train_accuracy_correct += correct
            total_train_accuracy_total += total

        ## Set model to eval mode
        model.eval()
        
        for batch in tqdm(val_dataloader):            

            ## ...
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            ## ...
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            ## Loss and Accuracy
            correct, total = calculate_mlm_scores(outputs, labels)
            loss = outputs.loss
        
            ## Tracker for validation loss and accuracy
            total_val_loss += loss.item()
            total_val_accuracy_correct += correct
            total_val_accuracy_total += total

        ## Updating tracker for calculating loss & accuracy per epoch
        step_train += len(train_dataloader)
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_accuracy = total_train_accuracy_correct / total_train_accuracy_total
        
        tracker["train_steps"].append(step_train)
        tracker["train_loss_tracker"].append(avg_train_loss)
        tracker["train_accuracy_tracker"].append(avg_train_accuracy)

        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "train_accuracy": avg_train_accuracy, "train_step": step_train})
 
        step_val += len(val_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_accuracy = total_val_accuracy_correct / total_val_accuracy_total

        tracker["val_loss_tracker"].append(avg_val_loss)
        tracker["val_accuracy_tracker"].append(avg_val_accuracy)
        tracker["val_steps"].append(step_val)

        wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss, "val_accuracy": avg_val_accuracy, "val_step": step_val})

        ## Save model
        model.save_pretrained(MODEL_SAVE_PATH+f"/epoch_{epoch+1}", overwrite=True)

        ## End time
        END_TIME = time.time()
        TOTAL_TIME = (END_TIME - START_TIME) / 60
        tracker["train_time"].append(TOTAL_TIME)
        
        ## Print Statements
        print(f"Training Loss: {avg_train_loss}")
        print(f"Training Accuracy: {avg_train_accuracy}")
        print(f"Validation Loss: {avg_val_loss}")
        print(f"Validation Accuracy: {avg_val_accuracy}")
        print(f"Time for epoch {epoch+1}: {(TOTAL_TIME)} minutes")
    
    ## Close wandb
    wandb.finish()

    ## Save tracker as XLS
    df = pd.DataFrame(tracker)
    df.to_excel(f'{MODEL_SAVE_PATH}/tracker.xlsx', index=False)

if __name__ == '__main__':
    
    ## Setting seed values
    SEED_VALUE = 42
    random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed_all(SEED_VALUE)

    ## Hyperparams for pretraining
    MLM_PROB = 0.15

    ## Hyperparams for training
    TRAIN_BATCH_SIZE = 14
    EVAL_BATCH_SIZE = 14
    MAX_EPOCHS = 10
    LEARNING_RATE = 0.0003
    WEIGHT_DECAY = 0.01

    ## Print Learning Rate
    print(f'Learning Rate: {LEARNING_RATE}')
    print(f'Weight Decay: {WEIGHT_DECAY}')

    ## Choose model checkpoint
    MODEL_CHECKPOINT = "bert-base-cased"

    ## Set paths
    DATA_PATH = "/home/kaydin/data/dataset_1M.pkl"
    MODEL_PATH = "/netscratch2/kaydin/models/"

    ## Set device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    ## Print statement
    print("Opening up Pickle Dataset")

    ## Open and load the Pickle file
    with open(DATA_PATH, 'rb') as pickle_file:
        dataset = pickle.load(pickle_file)

    ## Print statement
    print("Dataset is opened")

    ## Download model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    ## Define collator
    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB)

    ## Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset["train"], collate_fn=collate_fn, batch_size=TRAIN_BATCH_SIZE)
    val_dataloader = torch.utils.data.DataLoader(dataset["validation"], collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE)

    ## Train model
    train(MODEL_CHECKPOINT, DEVICE, train_dataloader, val_dataloader, max_epochs=MAX_EPOCHS, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
