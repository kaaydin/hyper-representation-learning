import subprocess 
import numpy as np
import json 
import os 
from tqdm import tqdm
from torch import nn
import torch
from transformers import BertTokenizer, BertModel

def read_txt_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def generate_embeddings(input_path, embed_path, model_path):
    
    ## Load model & extract BERT backbone
    model = torch.load(os.path.join(model_path, "final.pt"), map_location=torch.device('cpu'))
    model = model.bert

    ## Load data 
    data = read_txt_file(input_path)

    ## Collect hidden_states
    hidden_states = []

    ## Generate hidden embeddings
    with torch.no_grad():  # Ensure no gradients are computed
        for text_input in data: 
            inputs = tokenizer(text_input, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_state = outputs.hidden_states
            hidden_state = hidden_state[1:]
            hidden_states.append(hidden_state)

    print("Hidden States Generated")
    
    for layer in range(12):
        layers = []
        for text in hidden_states:
            # Detach the tensor, remove batch dimension, and convert to numpy
            layer_tensor = text[layer].squeeze(0).detach().cpu().numpy()
            layers.append(layer_tensor)

        concatenated = np.concatenate(layers, axis=0)

        saving_path = os.path.join(embed_path, f"layer_{layer+1}.npy")
        np.save(saving_path, concatenated)

if __name__ == '__main__':
    PATH_INPUT_DATA = "/home/kaydin/similarity/data/input_embeddings_squad.txt"
    PATH_EMBEDDING = "/netscratch2/kaydin/embeddings_squad"
    PATH_MODELS = "/netscratch2/sschnydrig/squad/models/"

    ## List all folders in directory
    models = os.listdir(PATH_MODELS)

    ## Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    for model in tqdm(models):
        model_path = os.path.join(PATH_MODELS, model)
        working_path = os.path.join(PATH_EMBEDDING, model)

        if not os.path.exists(working_path):
            os.makedirs(working_path)

        print("Generating Embeddings")
        generate_embeddings(input_path = PATH_INPUT_DATA, embed_path = working_path, model_path = model_path)
    