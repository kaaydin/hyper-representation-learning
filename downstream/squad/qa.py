# %%
from utils import prepare_train_features, prepare_validation_features, postprocess_qa_predictions, initialize_weights, prepare_features, configure_training

# %% [markdown]
# ## 1. Load data

# %%
from datasets import load_dataset, load_metric, load_from_disk
datasets = load_from_disk('squad/train_val_test_data')

# use only the first 100 elements of  datasets["train"] for demonstration purposes
for key in datasets.keys(): datasets[key] = datasets[key].select(range(100))

# %% [markdown]
# ## 2. Preprocess data

# %%
# Tokenize and prepare features for each dataset split using helper functions defined in utils.py
tokenized_datasets = datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)
validation_features = prepare_features("validation", prepare_validation_features, datasets)
train_features = prepare_features("train", prepare_validation_features, datasets)
test_features = prepare_features("test", prepare_validation_features, datasets)

# %% [markdown]
# ## 4. Model training

# %%
import wandb
import csv
import os
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, TrainerCallback, default_data_collator, AutoTokenizer

# Define your options for head and init_type
head_options = ['standard', 'linear', 'mlp']
init_type_options = ['standard', 'he', 'xavier', 'uniform']

# Other parameters
model_checkpoint = "bert-base-cased"
learning_rate = 2e-5
weight_decay = 0.01
epochs = 5
batch_size = 16
SAVE_PATH = "/netscratch2/sschnydrig/squad/models/test"
# create save path if it doesn't exist
os.makedirs(SAVE_PATH, exist_ok=True)

# Function to configure and train the model
def train_model(head, init_type):
    model_name = f"bert-base-cased_lr_{learning_rate}_wd_{weight_decay}_head_{head}_init_{init_type}"
    save_model_path = f"{SAVE_PATH}/{model_name}"
    os.makedirs(save_model_path, exist_ok=True)

    # Configure your model and training
    model, training_args = configure_training(
        model_checkpoint, model_name, learning_rate, weight_decay, head, init_type, epochs, batch_size, save_model_path
    )

    # Initialize wandb
    wandb.login()
    wandb.init(project="Squad", entity="imp-language", 
        name=model_name,
        config={"learning_rate": learning_rate, "weight_decay": weight_decay, init_type: "init_type", head: "head", model_checkpoint: "model_checkpoint"
        }
    )

    from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, TrainerCallback, default_data_collator, AutoTokenizer
    import csv
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    data_collator = default_data_collator

    squad_v2 = False # flag to indicate whether to use SQuAD 2.0 or 1.1
    metric = load_metric("squad_v2" if squad_v2 else "squad")    

    class EvalAndSaveCallback(TrainerCallback):
        def __init__(self):
            self.train_losses = []
            self.eval_losses = []
            self.test_losses = []  # Added for test data
            self.train_f1_scores = []
            self.train_em_scores = []
            self.eval_f1_scores = []
            self.eval_em_scores = []
            self.test_f1_scores = []  # Added for test data
            self.test_em_scores = []  # Added for test data
            # Initialize CSV file with column names
            with open(f"{save_model_path}/evaluation_results.csv", "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Test Loss", "Train F1 Score", "Train EM Score", "Validation F1 Score", "Validation EM Score", "Test F1 Score", "Test EM Score"])  # Updated

        def set_trainer(self, trainer):
            self.trainer = trainer
        
        def on_epoch_end(self, args, state, control, **kwargs):
            # Evaluate on training data
            print("Evaluating on training data...")
            train_result = trainer.evaluate(eval_dataset=tokenized_datasets["train"])
            self.train_losses.append(train_result["eval_loss"])

            # Predict and compute metrics for training data
            train_predictions = trainer.predict(train_features)  # train_features should be prepared similarly to validation_features
            train_final_predictions = postprocess_qa_predictions(datasets["train"], train_features, train_predictions.predictions)
            train_formatted_predictions, train_references = self.format_predictions(train_final_predictions, datasets["train"])
            train_metrics = metric.compute(predictions=train_formatted_predictions, references=train_references)
            self.train_f1_scores.append(train_metrics["f1"])
            self.train_em_scores.append(train_metrics["exact_match"])

            # Evaluate on validation data
            print("Evaluating on validation data...")
            eval_result = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
            self.eval_losses.append(eval_result["eval_loss"])

            # Predict and compute metrics for validation data
            eval_predictions = trainer.predict(validation_features)
            eval_final_predictions = postprocess_qa_predictions(datasets["validation"], validation_features, eval_predictions.predictions)
            eval_formatted_predictions, eval_references = self.format_predictions(eval_final_predictions, datasets["validation"])
            eval_metrics = metric.compute(predictions=eval_formatted_predictions, references=eval_references)
            self.eval_f1_scores.append(eval_metrics["f1"])
            self.eval_em_scores.append(eval_metrics["exact_match"])

            # Evaluate on test data
            print("Evaluating on test data...")
            test_result = trainer.evaluate(eval_dataset=tokenized_datasets["test"])  # Added for test data
            self.test_losses.append(test_result["eval_loss"])  # Added for test data

            # Predict and compute metrics for test data
            test_predictions = trainer.predict(test_features)  # test_features should be prepared
            test_final_predictions = postprocess_qa_predictions(datasets["test"], test_features, test_predictions.predictions)
            test_formatted_predictions, test_references = self.format_predictions(test_final_predictions, datasets["test"])
            test_metrics = metric.compute(predictions=test_formatted_predictions, references=test_references)
            self.test_f1_scores.append(test_metrics["f1"])  # Added for test data
            self.test_em_scores.append(test_metrics["exact_match"])  # Added for test data

            # Save results to CSV
            with open(f"{save_model_path}/evaluation_results.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    state.epoch,
                    train_result["eval_loss"],
                    eval_result["eval_loss"],
                    test_result["eval_loss"],  # Added for test data
                    train_metrics["f1"],
                    train_metrics["exact_match"],
                    eval_metrics["f1"],
                    eval_metrics["exact_match"],
                    test_metrics["f1"],  # Added for test data
                    test_metrics["exact_match"]  # Added for test data
                ])
            
            # Log metrics to wandb
            wandb.log({
                "Epoch": state.epoch,
                "Train Loss": train_result["eval_loss"],
                "Validation Loss": eval_result["eval_loss"],
                "Test Loss": test_result["eval_loss"],
                "Train F1": train_metrics["f1"],
                "Train EM": train_metrics["exact_match"],
                "Validation F1": eval_metrics["f1"],
                "Validation EM": eval_metrics["exact_match"],
                "Test F1": test_metrics["f1"],
                "Test EM": test_metrics["exact_match"]
            })

        def format_predictions(self, predictions, dataset):
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
            references = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset]
            return formatted_predictions, references

            
    callback = EvalAndSaveCallback()
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[callback]
    )

    # Start training
    trainer.train()
    trainer.save_model(f"{save_model_path}/final_model")

    # Access the results
    print("Train losses:", callback.train_losses)
    print("Evaluation losses:", callback.eval_losses)
    print("Train F1 Scores:", callback.train_f1_scores)
    print("Train EM Scores:", callback.train_em_scores)
    print("Validation F1 Scores:", callback.eval_f1_scores)
    print("Validation EM Scores:", callback.eval_em_scores)

# Loop over all permutations of head and init_type
for head in head_options:
    for init_type in init_type_options:
        train_model(head, init_type)