from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(f"/netscratch2/sschnydrig/pretraining/models/seed_lr_0.0001_wd_0.01_seed_1")

import transformers
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.

pad_on_right = tokenizer.padding_side == "right"

squad_v2 = False # flag to indicate whether to use SQuAD 2.0 or 1.1

def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

n_best_size = 20
max_answer_length = 30

from tqdm.auto import tqdm
import collections
import numpy as np

def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions

def prepare_features(dataset_split, prepare_function, datasets):
    features = datasets[dataset_split].map(prepare_function, batched=True, remove_columns=datasets[dataset_split].column_names)
    features.set_format(type=features.format["type"], columns=list(features.features.keys()))
    return features

def format_predictions(predictions, dataset):
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
            references = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset]
            return formatted_predictions, references

import torch.nn.init as init
import torch.nn as nn
def initialize_weights(m, init_type="uniform"):
    if isinstance(m, nn.Linear):
        if init_type == "he":
            # He initialization
            init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            print("Applied he initialisation")
        elif init_type == "xavier":
            # Xavier initialization
            init.xavier_normal_(m.weight)
            print("Applied xavier initialisation")
        elif init_type == "uniform":
            # Uniform initialization
            init.uniform_(m.weight, -0.1, 0.1)  # You can adjust the range as needed
            print("Applied uniform initialisation")
        else:
            print("Applied default initialisation")

        if m.bias is not None:
            init.constant_(m.bias, 0)

import csv
import numpy as np
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, TrainerCallback, AutoTokenizer, BertForQuestionAnswering
from datasets import load_metric
import torch.nn as nn
import wandb
import torch

def configure_training(model_checkpoint, learning_rate, weight_decay, head, init_type, gradual_unfreezing, scheduling, epochs, batch_size, save_model_path):
    #model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    model = BertForQuestionAnswering.from_pretrained(model_checkpoint)

    # Check if CUDA is available and set the device
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(f"Using device: {device}")
    #model.to(device)

    if head == "linear":
        model.qa_outputs = nn.Sequential(
            nn.Linear(768, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 2)
        )
    elif head == "mlp":
        model.qa_outputs = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    elif head == "mlp_crazy":
        model.qa_outputs = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    model.qa_outputs.apply(lambda m: initialize_weights(m, init_type=init_type))

    if gradual_unfreezing:
        # Initially freeze all layers
        for name, param in model.named_parameters():
            if 'qa_outputs' not in name:  # Keep the classification head always unfrozen
                param.requires_grad = False

        print("Layers overview before starting training")        
        for name, param in model.named_parameters():
                        if param.requires_grad == False:
                            print(f"Frozen layer: {name}")
                        elif param.requires_grad == True:
                            print(f"Unfrozen layer: {name}")

    if scheduling == "warmup":
        args = TrainingArguments(
            output_dir=f"{save_model_path}/checkpoints",
            evaluation_strategy = "epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=10,  # Adjust this number as needed
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            #push_to_hub=True,
            #hub_token='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',  # Pass your token here
            report_to=["wandb"],
            save_strategy="epoch",
            fp16=True,
            warmup_ratio = 0.1,
            lr_scheduler_type="linear",  # Linear decay after warmup
        )
        print("Using warmup = 0.1")
    elif scheduling == "only_warmup":
        args = TrainingArguments(
            output_dir=f"{save_model_path}/checkpoints",
            evaluation_strategy = "epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=10,  # Adjust this number as needed
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            #push_to_hub=True,
            #hub_token='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',  # Pass your token here
            report_to=["wandb"],
            save_strategy="epoch",
            fp16=True,
            warmup_ratio = 0.1,
            lr_scheduler_type="constant_with_warmup",  # Linear decay after warmup
        )
        print("Using only warmup = 0.1")
    elif scheduling == "constant":
        print('constant')
        args = TrainingArguments(
            output_dir=f"{save_model_path}/checkpoints",
            evaluation_strategy = "epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=10,  # Adjust this number as needed
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            #push_to_hub=True,
            #hub_token='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',  # Pass your token here
            report_to=["wandb"],
            save_strategy="epoch",
            fp16=True,
            lr_scheduler_type="constant",  # Keep a constant learning rate
        )
        print("Using constant learning rate")
    elif scheduling == "default":
        args = TrainingArguments(
            output_dir=f"{save_model_path}/checkpoints",
            evaluation_strategy = "epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=10,  # Adjust this number as needed
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            #push_to_hub=True,
            #hub_token='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',  # Pass your token here
            report_to=["wandb"],
            save_strategy="epoch",
            fp16=True,
        )
        print("Using default scheduling")

    return model, args