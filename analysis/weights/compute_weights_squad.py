import torch
from torch import nn
import numpy as np
import os
from transformers import BertModel
from tqdm import tqdm

def generate_weights(model_path, saving_path):
    
    ## Load model & extract BERT backbone
    model = torch.load(os.path.join(model_path, "final.pt"), map_location=torch.device('cpu'))

    ## Extract weights from entire model
    weights = [param for _, param in model.named_parameters()]

    for i in range(13):
        if i != 12:
            start_idx = 5 + i * 16
            end_idx = start_idx + 16
        else:
            start_idx = 197 ## SQuAD does not have a pooler
            end_idx = len(weights)

        # Extract weights for each layer and vectorize them
        layer_weights = weights[start_idx:end_idx]
        vectorized_layer_weights = [w.detach().view(-1).numpy() for w in layer_weights]
        concatenated_weights = np.concatenate(vectorized_layer_weights)

        filename = f"bert_layer_{i+1}_weights.npy"
        save_path = os.path.join(saving_path, filename)
        np.save(save_path, concatenated_weights)

        print(f"Layer {i} weights saved as {save_path}")

if __name__ == '__main__':
    PATH_MODELS = "/netscratch2/sschnydrig/squad/models/"
    PATH_SAVE = "/netscratch2/kaydin/weights_squad"

    models = os.listdir(PATH_MODELS)

    for model in tqdm(models):
        model_path = os.path.join(PATH_MODELS, model)
        saving_path = os.path.join(PATH_SAVE, model)

        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        print(model)
        generate_weights(model_path = model_path, saving_path = saving_path)