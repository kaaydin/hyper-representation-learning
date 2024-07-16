import torch
from torch import nn
import numpy as np
import os
from transformers import BertModel
from tqdm import tqdm


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

def generate_weights(model_path, saving_path):
    
    ## Load model & extract BERT backbone
    model = SentimentClassifier(2, "bert-base-cased")
    model = torch.load(os.path.join(model_path, "epoch_5", "model.pt"), map_location=torch.device('cpu'))

    ## Extract weights from entire model
    weights = [param for _, param in model.named_parameters()]

    for i in range(13):
        if i != 12:
            start_idx = 5 + i * 16
            end_idx = start_idx + 16
        else:
            start_idx = 199
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
    PATH_MODELS = "/netscratch2/kaydin/sst"
    PATH_SAVE = "/netscratch2/kaydin/weights"

    models = os.listdir(PATH_MODELS)

    for model in tqdm(models):
        model_path = os.path.join(PATH_MODELS, model)
        saving_path = os.path.join(PATH_SAVE, model)

        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        print(model)
        generate_weights(model_path = model_path, saving_path = saving_path)