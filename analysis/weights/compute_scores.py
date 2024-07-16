import numpy as np 
from scipy.spatial.distance import cosine
from collections import defaultdict
from tqdm import tqdm
import os
import json

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def list_subfolders_with_prefix(directory, prefix):
    # List to store the names of the matching subfolders
    subfolders = []

    # Check each item in the directory
    for item in os.listdir(directory):
        # Construct full path
        full_path = os.path.join(directory, item)
        
        # Check if the item is a directory and starts with the prefix
        if os.path.isdir(full_path) and item.startswith(prefix):
            subfolders.append(item)

    return subfolders

def compute_scores(sub1, sub2, history):

    # Load weights from layer
    w1 = np.load(sub1)
    w2 = np.load(sub2)

    ## Compute cosine similarity, distance, and l2 norm
    cos_dist = cosine(w1, w2)
    history["cos_dist"].append(float(cos_dist))  # Convert to native float
    history["cos_sim"].append(1 - float(cos_dist))  # Convert to native float
    history["l2_norm"].append(float(np.linalg.norm(w1 - w2)))  # Convert to native float

    ## Normalize weights
    w1_norm = normalize_vector(w1)
    w2_norm = normalize_vector(w2)

    ## Compute cosine similarity, distance, and l2 norm for normalized weights
    cos_dist_norm = cosine(w1_norm, w2_norm)
    history["cos_dist_norm"].append(float(cos_dist_norm))  # Convert to native float
    history["cos_sim_norm"].append(1 - float(cos_dist_norm))  # Convert to native float
    history["l2_norm_norm"].append(float(np.linalg.norm(w1_norm - w2_norm)))  # Convert to native float


if __name__ == "__main__":
    PATH_WEIGHTS = "/netscratch2/kaydin/weights_squad"
    PATH_SAVE = "/home/kaydin/weight-analysis"
    MODEL_ZOO_TYPE = "continued" ## "seed", "scratch" 

    history = defaultdict(list)
    subfolders = list_subfolders_with_prefix(PATH_WEIGHTS, MODEL_ZOO_TYPE)

    for i, subfolder1 in enumerate(subfolders):
            print(f"Main Loop: {i+1} / {len(subfolders)}")
            for k in tqdm(range(i + 1, len(subfolders))):
                subfolder2 = subfolders[k]

                for j in range(13):
                    history["Model 1"].append(subfolder1)
                    history["Model 2"].append(subfolder2)
                    history["Layer"].append(j+1) 

                    sub1 = os.path.join(PATH_WEIGHTS, subfolder1, f"bert_layer_{j+1}_weights.npy")
                    sub2 = os.path.join(PATH_WEIGHTS, subfolder2, f"bert_layer_{j+1}_weights.npy")

                    compute_scores(sub1, sub2, history)
    
    ## Save scores as json
    save_path = os.path.join(PATH_SAVE, f"{MODEL_ZOO_TYPE}_squad.json")
    with open(save_path, 'w') as outfile:
        json.dump(history, outfile)