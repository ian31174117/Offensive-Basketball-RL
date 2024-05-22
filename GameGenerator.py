import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import argparse
import CustomVisualizer as CV
import copy
from sklearn.metrics.pairwise import cosine_similarity

from DefenseTransformer import *
from OffenseTransformer import *
from SiameseNetwork import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Defense Transformer Model')
    parser.add_argument('--task', type=str, required=True, help='Specify the task, can be generate or judge')
    parser.add_argument('--num_cases', type=int, default=5, help='Number of cases to generate')
    args = parser.parse_args()
    num_cases = 5
    if args.num_cases != None:
        num_cases = args.num_cases

    offense_path = "./data/offense_model.pth"
    offense_model = OffenseTransformer()
    offense_model.load_state_dict(torch.load(offense_path))
    offense_dataset = OffensiveBasketballDataset("./data/50Real.npy")
    offense_dataloader = DataLoader(offense_dataset, batch_size=1, shuffle=True)
    
    defense_path = "./data/trained_model_with_mask_eighty.pth"
    defense_model = DefenseTransformer()
    defense_model.load_state_dict(torch.load(defense_path))
    defense_dataset = BasketballDataset("./data/50Real.npy")
    defense_dataloader = DataLoader(defense_dataset, batch_size=1, shuffle=True)

    test_source = torch.FloatTensor(offense_dataset.test_source)
    samples = copy.deepcopy(test_source)
    tmp = copy.deepcopy(test_source.numpy())
    original_traj = copy.deepcopy(test_source.numpy())
    defense_tmp_targets = torch.FloatTensor(np.zeros((1,49,10)))
    offense_tmp_targets = torch.FloatTensor(np.zeros((1,49,15)))
    with torch.no_grad():
        for _ in range(num_cases):
            idx = np.random.randint(0,len(test_source))
            if args.task == "generate":
                CV.visualize_game(tmp[idx], anim_path=f"./fully_generated/fully_generated_game_{idx}_sample.gif", title = "Sample Game")
            for i in range(48):
                defense_action = defense_model(samples[0:1,:,:],defense_tmp_targets[0:1,0:i+1,:])
                offense_action = offense_model(samples[0:1,:,:],offense_tmp_targets[0:1,0:i+1,:])
                defense_tmp_targets[0:1,i+1:i+2,:] = defense_action[0:1,i:i+1,:]
                offense_tmp_targets[0:1,i+1:i+2,:] = offense_action[0:1,i:i+1,:]
                tmp[idx:idx+1,i+1,13:23] = tmp[idx:idx+1,i,13:23] + defense_action[0:1,i,:].numpy()
                tmp[idx:idx+1,i+1,3:13] = tmp[idx:idx+1,i,3:13] + offense_action[0:1,i,0:10].numpy()
                ballholder_id = 0
                cur_val = 0.0
                for k in range(5):
                    if offense_action[0:1,i:i+1,k+10] > cur_val:
                        cur_val = offense_action[0:1,i:i+1,k+10]
                        ballholder_id = k
                tmp[idx:idx+1,i+1,0] = tmp[idx:idx+1,i,3+ballholder_id*2]
                tmp[idx:idx+1,i+1,1] = tmp[idx:idx+1,i,4+ballholder_id*2]

                samples = torch.FloatTensor(tmp)
                #print(f"frame {i} done")
            if args.task == "generate":
                CV.visualize_game(tmp[idx], anim_path=f"./fully_generated/fully_generated_game_{idx}.gif", title = "Fully Generated Game")
            if args.task == "judge":
                siamese_model = SiameseNetwork(49*23, 128, 64)
                siamese_model.load_state_dict(torch.load("./data/trajectory_similarity_model.pth"))
                output1, output2 = siamese_model(torch.FloatTensor(original_traj[idx]).view(-1, 49*23), torch.FloatTensor(tmp[idx]).view(-1, 49*23))

                similarity_score = F.cosine_similarity(output1, output2, dim=1)
                print(f"Similarity score for game {idx}: {similarity_score.item()}")

