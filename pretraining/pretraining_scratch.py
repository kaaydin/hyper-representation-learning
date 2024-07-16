# %%
"""
!pip install git+https://github.com/huggingface/transformers.git
!pip install datasets
!pip install huggingface-hub
!pip install nltk
!pip install pytorch-lightning
"""

# %%
import time
start_time = time.time()

import nltk
import random
import logging

#nltk.download("punkt")

BLOCK_SIZE = 256  # Maximum number of tokens in an input sample
NSP_PROB = 0.50  # Probability that the next sentence is the actual ne
SHORT_SEQ_PROB = 0.1  # Probability of generating shorter sequences to minimize the mismatch between pretraining and fine-tuning.
MAX_LENGTH = 512  # Maximum number of tokens in an input sample after padding

MLM_PROB = 0.2  # Probability with which tokens are masked in MLM

TRAIN_BATCH_SIZE = 16  # Batch-size for pretraining the model on
MAX_EPOCHS = 10  # Maximum number of epochs to train the model for
MODEL_CHECKPOINT = "bert-base-cased"  # Name of pretrained model from 🤗 Model Hub

random.seed(420)
from datasets import load_from_disk
tokenized_dataset = load_from_disk("/netscratch2/sschnydrig/pretraining/data/tokenized_dataset")
#tokenized_dataset = load_from_disk("/netscratch2/kaydin/data_sven/tokenized_dataset_2")

"""
# Define a function to filter out examples with more than 512 tokens
def filter_long_examples(example):
    return len(example['input_ids']) <= 512

# Apply the filter to both the train and validation datasets
tokenized_dataset["train"] = tokenized_dataset["train"].filter(filter_long_examples)
tokenized_dataset["validation"] = tokenized_dataset["validation"].filter(filter_long_examples)

# Assuming you have already applied the filter as shown in the previous answer
# Print the size of the train dataset
print("Size of the train dataset:", len(tokenized_dataset["train"]))
# Print the size of the validation dataset
print("Size of the validation dataset:", len(tokenized_dataset["validation"]))
"""

# %%
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers import BertForPreTraining, Trainer, TrainingArguments
import csv

class CSVLoggerCallback(TrainerCallback):
    """A callback that logs the training and evaluation results to a CSV file."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._has_written = False
        self.train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Save the train loss for writing in on_evaluate
        if 'loss' in logs:
            self.train_loss = logs['loss']

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # If the CSV file doesn't exist, write the header
        if not self._has_written:
            with open(self.csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                # Add mlm_accuracy and nsp_accuracy to the header
                writer.writerow(['epoch', 'step', 'train_loss', 'eval_loss'])
                self._has_written = True

        # Log the metrics along with the last recorded train_loss
        with open(self.csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            # Add mlm_accuracy and nsp_accuracy to the logged metrics
            writer.writerow([
                state.epoch,
                state.global_step,
                self.train_loss,
                metrics['eval_loss'] if 'eval_loss' in metrics else None,
            ])

        # Reset the training loss at the end of each epoch
        self.train_loss = None

import matplotlib.pyplot as plt
import pandas as pd
def plot_metrics(log_path, title, save_path):
    # Read the CSV file containing the logs
    df = pd.read_csv(log_path)

    plt.figure(figsize=(10, 6))

    # Define different shades of green for the plots
    colors = ['green', 'limegreen', 'darkgreen']  # You can adjust these colors as needed

    # Check which columns are available and plot them in shades of green
    if 'train_loss' in df.columns:
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color=colors[0])

    if 'eval_loss' in df.columns:
        plt.plot(df['epoch'], df['eval_loss'], label='Eval Loss', color=colors[1])
    
    if 'eval_accuracy' in df.columns:
        plt.plot(df['epoch'], df['eval_accuracy'], label='Eval Accuracy', color=colors[2])

    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title(title)
    plt.legend()
    plt.grid(False)
    plt.savefig(save_path)
    plt.show()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

from transformers import DataCollatorForLanguageModeling
# Create the data collator
collater = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,  # Masked Language Model (MLM) objective
    mlm_probability=MLM_PROB,  # Probability of masking tokens (adjust as needed)
)

# %%
from transformers import BertForPreTraining, Trainer, TrainingArguments, BertConfig
import torch
import os
import wandb
from transformers import AdamW
from transformers.integrations import WandbCallback  # Import WandbCallback

learning_rates = [1e-4]
weight_decays = [1e-1]
schedulers = ["warmup"]
wrs = [0.2]

run_name = "pretraining_scratch"

PATH = f"/netscratch2/sschnydrig/pretraining/models/{run_name}"
#tokenizer.save_pretrained(f'{PATH}/tokenizer')
for lr in learning_rates:
    for wd in weight_decays:
        for scheduler in schedulers:
            for wr in wrs:
                # Start timing
                start_time = time.time()
                print("---------------Training started---------------")
                import torch
                torch.cuda.empty_cache()

                # Check if MPS device is available and use it
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Using device: {device}")

                # Define your model name based on learning rate (lr) and weight decay (wd)
                MODEL_NAME = f"bert_lr_{lr}_wd_{wd}"

                SAVE_PATH = f"{PATH}/{MODEL_NAME}"
                os.makedirs(SAVE_PATH, exist_ok=True)

                # Initialize W&B run
                wandb.init(project=f"{run_name}", entity="imp-language",
                name=MODEL_NAME,
                config={"learning_rate": lr, "weight_decay": wd, "scheduler": scheduler, "model_checkpoint": MODEL_CHECKPOINT,
                }
                )
                
                # Initialize BERT model from scratch
                config = BertConfig()
                model = BertForPreTraining(config)
                
                if scheduler == "default":
                # Define the training arguments
                    training_args = TrainingArguments(
                        output_dir=f"{SAVE_PATH}/checkpoints",
                        overwrite_output_dir=True,
                        num_train_epochs=MAX_EPOCHS,
                        per_device_train_batch_size=TRAIN_BATCH_SIZE,
                        per_device_eval_batch_size=TRAIN_BATCH_SIZE,
                        save_strategy="epoch",
                        evaluation_strategy="epoch",
                        logging_strategy="epoch",
                        report_to="wandb",  # Enable logging to wandb
                        learning_rate=lr,
                        weight_decay=wd,  # Weight decay if we apply some
                        fp16=True,
                        dataloader_num_workers=8,
                    )
                elif scheduler == "warmup":
                        # Define the training arguments
                        training_args = TrainingArguments(
                            output_dir=f"{SAVE_PATH}/checkpoints",
                            overwrite_output_dir=True,
                            num_train_epochs=MAX_EPOCHS,
                            per_device_train_batch_size=TRAIN_BATCH_SIZE,
                            per_device_eval_batch_size=TRAIN_BATCH_SIZE,
                            save_strategy="epoch",
                            evaluation_strategy="epoch",
                            logging_strategy="epoch",
                            report_to="wandb",  # Enable logging to wandb
                            learning_rate=lr,
                            weight_decay=wd,  # Weight decay if we apply some
                            fp16=True,
                            dataloader_num_workers=8,
                            warmup_ratio = wr,
                            lr_scheduler_type="linear",  # Linear decay after warmup
                        )
                elif scheduler == "constant":
                        training_args = TrainingArguments(
                            output_dir=f"{SAVE_PATH}/checkpoints",
                            overwrite_output_dir=True,
                            num_train_epochs=MAX_EPOCHS,
                            per_device_train_batch_size=TRAIN_BATCH_SIZE,
                            per_device_eval_batch_size=TRAIN_BATCH_SIZE,
                            save_strategy="epoch",
                            evaluation_strategy="epoch",
                            logging_strategy="epoch",
                            report_to="wandb",  # Enable logging to wandb
                            learning_rate=lr,
                            weight_decay=wd,  # Weight decay if we apply some
                            lr_scheduler_type='constant',  # Use a constant learning rate
                            fp16=True,
                            dataloader_num_workers=8,
                        )
                
                csv_logger = CSVLoggerCallback(csv_path=f'{SAVE_PATH}/{MODEL_NAME}_logs.csv')

                # Initialize the Trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=collater,
                    train_dataset=tokenized_dataset["train"],
                    eval_dataset=tokenized_dataset["validation"],
                    tokenizer=tokenizer,
                    callbacks=[csv_logger],  # Add the CSV logger callback
                )

                # Start training
                trainer.train()

                # Evaluation
                #trainer.evaluate()

                # Save the final model
                trainer.save_model(f"{SAVE_PATH}/final")

                # Call the plotting function
                plot_metrics(
                log_path=f'{SAVE_PATH}/{MODEL_NAME}_logs.csv',
                title=f"Training and Evaluation Metrics for {MODEL_NAME}",
                save_path=f"{SAVE_PATH}/{MODEL_NAME}_metrics.png"
                )

                # Finish the wandb run
                wandb.finish()

                # End timing
                end_time = time.time()

                # Calculate total training time
                total_training_time = (end_time - start_time) / 60 / 60
                print("---------------------------------------------------")
                print(f"Total training time: {total_training_time} hours")
                # %%