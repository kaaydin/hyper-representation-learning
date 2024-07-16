import subprocess 
import numpy as np
import json 
import os 
from tqdm import tqdm
from torch import nn
import torch
from transformers import BertTokenizer, BertModel

class SentimentClassifier(nn.Module):

    def __init__(self, n_classes, pretrained_model_path):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        
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

def read_txt_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def generate_embeddings(input_path, embed_path, model_path):
    
    ## Load model & extract BERT backbone
    model = SentimentClassifier(2, "bert-base-cased")
    model = torch.load(os.path.join(model_path, "epoch_5", "model.pt"), map_location=torch.device('cpu'))
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
    PATH_INPUT_DATA = "/home/kaydin/similarity/input_embeddings_sst.txt"
    PATH_EMBEDDING = "/netscratch2/kaydin/embeddings_sst"
    PATH_MODELS = "/netscratch2/kaydin/sst"

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
    