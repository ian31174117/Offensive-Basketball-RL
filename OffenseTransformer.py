import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import argparse
import CustomVisualizer as CV
import copy

class OffensiveBasketballDataset(Dataset):
    def __init__(self, filename):
        self.test_source = []
        self.test_target = []
        self.test_target_y = []
        tmp_sources = np.load(filename)
        
        self.timescale = tmp_sources.shape[1] #50 in 50Real.npy
        self.len = len(tmp_sources)
        #print(tmp_sources.shape)
        tmp_sources = tmp_sources[:,:,:,:3]
        tmp_sources = tmp_sources.reshape(tmp_sources.shape[0],tmp_sources.shape[1],-1)
        tmp_sources = tmp_sources[:,:,[0,1,2,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31]]
        #------------------------------ball--offensive--defensive------------------------------------
        #print(tmp_sources.shape)
        self.sources = []
        self.targets = []
        self.targets_y = []
        for i in range(self.len):
            to_append_sources = []
            to_append_targets = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
            for j in range(tmp_sources[i].shape[0]-1):
                to_append_sources.append(tmp_sources[i][j])
                ballholder_id = 0
                cur_min_dist = 100000.0
                for k in range(5):
                    cur_dist = (tmp_sources[i][j+1][k*2+3] - tmp_sources[i][j][0])**2 + (tmp_sources[i][j+1][k*2+4] - tmp_sources[i][j][1])**2
                    if cur_dist < cur_min_dist:
                        cur_min_dist = cur_dist
                        ballholder_id = k
                one_hot = [0.0,0.0,0.0,0.0,0.0]
                one_hot[ballholder_id] = 1.0
                tmp = (tmp_sources[i][j+1]-tmp_sources[i][j])[3:13]
                tmp = np.concatenate((tmp,one_hot))
                if(j==0) and (i==0):
                    print("tmp shape:",tmp.shape)
                    print("tmp:",tmp)
                to_append_targets.append(tmp)
            if i<=self.len * 0.9:
                self.sources.append(to_append_sources)
                self.targets.append(to_append_targets[:-1])
                self.targets_y.append(to_append_targets[1:])
            else:
                to_append_targets = to_append_targets[:-1]
                self.test_source.append(to_append_sources)
                self.test_target.append(to_append_targets)
                self.test_target_y.append(to_append_targets[1:])
        self.sources = np.array(self.sources)
        self.targets = np.array(self.targets)
        self.targets_y = np.array(self.targets_y)
        print("sources:", self.sources.shape)
        print("targets:", self.targets.shape)
        #self.sources = torch.tensor(self.sources, dtype=torch.float64)
        #self.targets = torch.tensor(self.targets, dtype=torch.float64)
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, index):
        return torch.FloatTensor(self.sources[index]), torch.FloatTensor(self.targets[index]), torch.FloatTensor(self.targets_y[index])

class OffenseTransformer(nn.Module):
    def __init__(self, input_dim = 23, output_dim =15, hidden_dim=128, num_layers=2, nhead=2):
        super(OffenseTransformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.outpur_fc = nn.Linear(output_dim, hidden_dim)

        self.transformer = nn.Transformer(d_model=hidden_dim, nhead = nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first= True)
        
        self.fc = nn.Linear(hidden_dim,output_dim)

    def forward(self, src, tgt):
        src = self.input_fc(src)
        tgt = self.outpur_fc(tgt)
        
        src_mask = self.transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        #for i in range(tgt_mask.shape[0]): tgt_mask[i][i] = float("-inf")
        #print("src:",src_mask)
        #print("tgt:",tgt_mask)
        

        output = self.transformer(src, tgt, src_mask = src_mask, tgt_mask = tgt_mask)

        output = self.fc(output)
        return output

def train(model, DL, save_path, plot_path, lr = 3e-5, epochs = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #samples, targets = next(iter(DL))
    #output = model(samples, targets)
    #print(output.shape)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    

    for epoch in range(epochs):
        model.train()

        iter_count = 0
        total_loss = 0.0
        for samples, targets, targets_y in DL:
            iter_count += 1
            if(iter_count == len(DL) -1 ): break
            samples, targets, targets_y = samples.to(device), targets.to(device), targets_y.to(device)

            # Forward pass
            output = model(samples, targets)

            #nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Compute the loss
            if(iter_count ==5):
                print("output shape", output.shape)
                print("targets shape:", targets.shape)
            loss = criterion(output, targets_y)

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        average_loss = total_loss / len(DL)
        losses.append(average_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {average_loss}")
    
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig(plot_path)
    #plt.show()

    torch.save(model.state_dict(), save_path)
    print("Training complete.")
    return


if __name__ == "__main__":
    filename ="./data/50Real.npy"
    OBD = OffensiveBasketballDataset(filename)
    ODL = DataLoader(OBD, batch_size = 4, shuffle = True)
    model = OffenseTransformer()

    parser = argparse.ArgumentParser(description='Defense Transformer Model')
    parser.add_argument('--task', type=str, required=True, help='Specify the task, can be train or test')
    args = parser.parse_args()

    save_path = "./data/offense_model.pth"

    if args.task == "train":
        plot_path = "./data/offense_trained_loss.png"
        train(model, ODL, save_path, plot_path, epochs = 50)
        