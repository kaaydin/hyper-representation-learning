import json
from collections import defaultdict
from tqdm import tqdm
import os
import numpy as np

from scoring import *


def compute_scores(sub1, sub2, history):

    ## Load embeddings
    rep1 = np.load(sub1)
    rep2 = np.load(sub2)

    ## Transpose shapes
    rep1 = rep1.T
    rep2 = rep2.T

    ## Center each row
    rep1 = rep1 - rep1.mean(axis=1, keepdims=True)
    rep2 = rep2 - rep2.mean(axis=1, keepdims=True)

    ## Normalize each representation
    rep1 = rep1 / np.linalg.norm(rep1)
    rep2 = rep2 / np.linalg.norm(rep2)

    ## CCA Decomposition
    cca_u, cca_rho, cca_vh, transformed_rep1, transformed_rep2 = cca_decomp(rep1, rep2)

    ## Calculate all scores
    history["PWCCA"].append(pwcca_dist(rep1, cca_rho, transformed_rep1))
    history["mean_sq_cca_corr"].append(mean_sq_cca_corr(cca_rho))
    history["mean_cca_corr"].append(mean_cca_corr(cca_rho))
    
    lin_cka_sim = lin_cka_dist(rep1, rep2)
    history["CKA"].append(lin_cka_sim)

    lin_cka_sim = lin_cka_prime_dist(rep1, rep2)
    history["CKA'"].append(lin_cka_sim)

    history["Procrustes"].append(procrustes(rep1, rep2))


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
    

if __name__ == '__main__':

    PATH_EMBEDDING = "/netscratch2/kaydin/embeddings_squad"
    MODEL_ZOO_TYPE = "scratch" ## "seed", "scratch"
    SAVE_PATH_SCORE = "/home/kaydin/similarity"

    print(PATH_EMBEDDING)

    history = defaultdict(list)
    subfolders = list_subfolders_with_prefix(PATH_EMBEDDING, MODEL_ZOO_TYPE)

    for i, subfolder1 in enumerate(subfolders):
        print(f"Main Loop: {i+1} / {len(subfolders)}") 
        for k in tqdm(range(i + 1, len(subfolders))):
        
            subfolder2 = subfolders[k]
            
            for j in range(12):    
                history["Model 1"].append(subfolder1)
                history["Model 2"].append(subfolder2)
                history["Layer"].append(str(j+1))
                
                sub1 = os.path.join(PATH_EMBEDDING, subfolder1, f"layer_{j+1}.npy")
                sub2 = os.path.join(PATH_EMBEDDING, subfolder2, f"layer_{j+1}.npy")

                compute_scores(sub1, sub2, history)
    
    ## Save scores as json
    save_path = os.path.join(SAVE_PATH_SCORE, f"{MODEL_ZOO_TYPE}_squad.json")
    with open(save_path, 'w') as outfile:
        json.dump(history, outfile)