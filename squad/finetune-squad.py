# %%
from utils import prepare_train_features, prepare_validation_features, postprocess_qa_predictions, initialize_weights, prepare_features, configure_training, format_predictions

# %% [markdown]
# ## 1. Load data

# %%
from datasets import load_dataset, load_metric, load_from_disk
datasets = load_from_disk('train_val_test_data')

# use only the first 100 elements of  datasets["train"] for demonstration purposes
#for key in datasets.keys(): datasets[key] = datasets[key].select(range(5_000))

# dataset_size = 32
# size_name = int(round(dataset_size / 1_000))
# datasets["train"] = datasets["train"].select(range(dataset_size))

# datasets["validation"] = datasets["validation"].select(range(dataset_size))

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

# previous run
# learning_rates = [1e-4, 5e-5, 3e-4]
# weight_decays = [1e-4]
# scheduling_options = ["only_warmup", "warmup", "constant"]
# gradual_unfreezing_options = [False]
# head_options = ["linear"]

# model_checkpoints = [
#                     f"{path}/bert_lr_5e-05_wd_0.0001/final", 
#                     f"{path}/bert_lr_3e-05_wd_0.0001/final", 
#                     ]

# run 20k diversity
""" learning_rates = [1e-4, 3e-4, 1e-3, 3e-3]
weight_decays = [1e-1, 1e-2, 1e-3]
scheduling_options = ["warmup"]
gradual_unfreezing_options = [False]
head_options = ["mlp", "mlp_crazy"] """

path = "/netscratch2/sschnydrig/pretraining/models"

import os
# create list with all model checkpoints
model_checkpoints = os.listdir(path)
model_checkpoints = [x for x in model_checkpoints if not "archive" in x and not "seed" in x]
model_checkpoints.append("bert-base-cased")


model_checkpoints = [
    'seed_lr_0.0001_wd_0.01_seed_1', 
    ]


#model_checkpoints = ["bert-base-cased"]


# Variable hyperparameters
learning_rates = [0.0001, 0.00001]
weight_decays = [0.01, 0.0001]

learning_rates = [0.0001]
weight_decays = [0.01]
scheduling_options = ["warmup"]
optimisers = ["adamw"]
head_options = ["mlp"]
gradual_unfreezing_options = [False]

# Fixed hyperparameters
epochs = 5
batch_size = 32
init_type = "he"

run_name = f"Squad"
import time
# Function to configure and train the model
def train_model(learning_rate, weight_decay, head, init_type, gradual_unfreezing, scheduling, model_checkpoint, optimiser):
    # Start timing
    start_time = time.time()
    print("---------------Training started---------------")

    print("model_checkpoint", model_checkpoint)

    SEED = False
    if "seed" in model_checkpoint:
        SEED = True
        model_checkpoint = f"{path}/{model_checkpoint}"
        print("model checkpoint:", model_checkpoint)
        name = model_checkpoint.split('/')[-1]
        print("name:", name)
    elif "scratch" in model_checkpoint or "continued" in model_checkpoint:
        model_checkpoint = f"{path}/{model_checkpoint}/final"
        name = model_checkpoint.split('/')[-2]
    elif "netscratch2" in model_checkpoint and "checkpoints" in model_checkpoint:
        model_checkpoint = f"{path}/{model_checkpoint}/final"
        name = model_checkpoint.split('/')[-3]
    elif "bert-base-cased" in model_checkpoint:
        model_checkpoint = "bert-base-cased"
        name = "bert-base-cased"

    print("model_checkpoint", model_checkpoint)
    print("name", name)

    SAVE_PATH = f"/netscratch2/sschnydrig/squad/models"
    os.makedirs(SAVE_PATH, exist_ok=True)

    import torch
    torch.cuda.empty_cache()

    def format_float(value):
        # Format the float with enough precision, then strip trailing zeros and the decimal point if it's not needed
        return "{:.10f}".format(value).rstrip('0').rstrip('.')
    model_name = f"{name}_squad_lr_{format_float(learning_rate)}_wd_{format_float(weight_decay)}_dropouts_0.25"

    save_model_path = f"{SAVE_PATH}/{model_name}"
    # Check if the model already exists
    if os.path.exists(save_model_path):
        print(f"Model {model_name} already exists. Skipping training.")
        return
    
    os.makedirs(save_model_path, exist_ok=True)

   # Configure your model and training
    model, training_args = configure_training(
        model_checkpoint, learning_rate, weight_decay, head, init_type, gradual_unfreezing, scheduling, epochs, batch_size, save_model_path
    )

    # Initialize wandb
    wandb.init(project=f"{run_name}", entity="imp-language", 
        name=model_name,
        config={"learning_rate": learning_rate, "weight_decay": weight_decay, init_type: "init_type", head: "head", "model_checkpoint": model_checkpoint,
                "scheduling": scheduling, "gradual_unfreezing": gradual_unfreezing, "optimiser": optimiser, 
        }
    )

    from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, TrainerCallback, default_data_collator, AutoTokenizer
    import csv
    if SEED == True:
        tokenizer = AutoTokenizer.from_pretrained(f"/netscratch2/sschnydrig/pretraining/models/{name}")
        print("Using seed tokenizer")
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        print("Using bert-base-cased tokenizer")

    data_collator = default_data_collator

    squad_v2 = False # flag to indicate whether to use SQuAD 2.0 or 1.1
    metric = load_metric("squad_v2" if squad_v2 else "squad")

    class EvalAndSaveCallback(TrainerCallback):
        def __init__(self):
            self.learning_rates = []
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
                writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Test Loss", "Train F1 Score", "Train EM Score", "Validation F1 Score", "Validation EM Score", "Test F1 Score", "Test EM Score", "Learning Rate"])  # Updated

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
            train_formatted_predictions, train_references = format_predictions(train_final_predictions, datasets["train"])
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
            eval_formatted_predictions, eval_references = format_predictions(eval_final_predictions, datasets["validation"])
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
            test_formatted_predictions, test_references = format_predictions(test_final_predictions, datasets["test"])
            test_metrics = metric.compute(predictions=test_formatted_predictions, references=test_references)
            self.test_f1_scores.append(test_metrics["f1"])  # Added for test data
            self.test_em_scores.append(test_metrics["exact_match"])  # Added for test data

            current_lr = trainer.optimizer.param_groups[0]["lr"]
            self.learning_rates.append(current_lr)
            print(f"Current learning rate: {current_lr}")

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
                    test_metrics["exact_match"],  # Added for test data
                    current_lr,  # Add current learning rate
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
                "Test EM": test_metrics["exact_match"],
                "Learning Rate": current_lr,  # Log current learning rate
            })

            # Gradual unfreezing logic
            if gradual_unfreezing:
                # Unfreeze layers 8, 9, 10, 11 after the first epoch
                if state.epoch == 1:
                    for i, layer in enumerate(model.bert.encoder.layer):
                        if i in [0, 1, 2, 3]:
                            for name, param in layer.named_parameters():
                                param.requires_grad = True
                    
                # Unfreeze layers 4, 5, 6, 7 after the second epoch
                elif state.epoch == 2:
                    for i, layer in enumerate(model.bert.encoder.layer):
                        if i in [4, 5, 6, 7]:
                            for name, param in layer.named_parameters():
                                param.requires_grad = True
                                #print(f"Unfroze layer {i}, parameter {name}")
        
                # Unfreeze all layers after the third epoch
                elif state.epoch == 3:
                    for name, param in model.named_parameters():
                        param.requires_grad = True
                
                if state.epoch in [1, 2, 3]:
                    print(f"Layers overview in epoch {state.epoch}")
                    for name, param in model.named_parameters():
                        if param.requires_grad == False:
                            print(f"Frozen layer: {name}")
                        elif param.requires_grad == True:
                            print(f"Unfrozen layer: {name}")

            # Save the entire model at the end of each epoch
            torch.save(model, os.path.join(save_model_path, f"epoch_{int(state.epoch)}.pt"))
            
    callback = EvalAndSaveCallback()

    if optimiser != "adamw":
        if optimiser == "nesterov":
            from torch.optim import SGD
            # Define SGD optimizer with momentum
            optimizer = SGD(model.parameters(), lr=training_args.learning_rate, momentum=0.9, nesterov=True, weight_decay=training_args.weight_decay)
        elif optimiser == "sgd":
            from torch.optim import SGD
            # Define SGD optimizer with momentum
            optimizer = SGD(model.parameters(), lr=training_args.learning_rate, momentum=0.9, nesterov=False, weight_decay=training_args.weight_decay)
        elif optimiser == "rmsprop":
            from torch.optim import RMSprop
            # Define RMSprop optimizer
            # Note: Adjust the alpha and weight_decay parameters as per your requirement
            optimizer = RMSprop(model.parameters(), lr=training_args.learning_rate, alpha=0.99, weight_decay=training_args.weight_decay)
        elif optimiser == "nadam":
            from torch.optim import Nadam
            # Define Nadam optimizer
            optimizer = Nadam(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
        elif optimiser == "adagrad":
            from torch.optim import Adagrad

            # Define Adagrad optimizer
            optimizer = Adagrad(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
        elif optimiser == "lamb":
            from pytorch_lamb import Lamb

            # Define LAMB optimizer
            optimizer = Lamb(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
        
        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            optimizers=(optimizer, None),  # Custom optimizer, None for scheduler (handled by Trainer)
            callbacks=[callback],
            )
    else:
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
    trainer.save_model(f"{save_model_path}/final")

    # Save the entire model
    torch.save(model, f"{save_model_path}/final.pt")

    # Access the results
    print("Train losses:", callback.train_losses)
    print("Evaluation losses:", callback.eval_losses)
    print("Train F1 Scores:", callback.train_f1_scores)
    print("Train EM Scores:", callback.train_em_scores)
    print("Validation F1 Scores:", callback.eval_f1_scores)
    print("Validation EM Scores:", callback.eval_em_scores)

    ## Close wandb
    wandb.finish()

    # End timing
    end_time = time.time()

    # Calculate total training time
    total_training_time = (end_time - start_time) / 60
    print("---------------------------------------------------")
    print(f"Total training time: {total_training_time} minutes")

# Loop over all permutations of head and init_type
#for head in head_options:
#    for init_type in init_type_options:
#        train_model(head, init_type)

for scheduling in scheduling_options:
    for weight_decay in weight_decays:
        for model_checkpoint in model_checkpoints:
            for gradual_unfreezing in gradual_unfreezing_options:
                for learning_rate in learning_rates:
                    for head in head_options:
                        for optimiser in optimisers:
                            train_model(learning_rate, weight_decay, head, init_type, gradual_unfreezing, scheduling, model_checkpoint, optimiser)