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

test_source = []
test_target = []

class BasketballDataset(Dataset):
    def __init__(self, filename):
        tmp_sources = np.load(filename)
        
        self.timescale = tmp_sources.shape[1] #50 in 50Real.npy
        self.len = len(tmp_sources)
        print(tmp_sources.shape)
        tmp_sources = tmp_sources[:,:,:,:3]
        tmp_sources = tmp_sources.reshape(tmp_sources.shape[0],tmp_sources.shape[1],-1)
        tmp_sources = tmp_sources[:,:,[0,1,2,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31]]
        print(tmp_sources.shape)
        self.sources = []
        self.targets = []
        for i in range(self.len):
            to_append_sources = []
            to_append_targets = []
            for j in range(tmp_sources[i].shape[0]-1):
                to_append_sources.append(tmp_sources[i][j])
                to_append_targets.append((tmp_sources[i][j+1]-tmp_sources[i][j])[13:23])
            if i<=self.len * 0.8:
                self.sources.append(to_append_sources)
                self.targets.append(to_append_targets)
            else:
                test_source.append(to_append_sources)
                test_target.append(to_append_targets)
        self.sources = np.array(self.sources)
        self.targets = np.array(self.targets)
        #print(self.sources.shape)
        #print(self.targets.shape)
        #self.sources = torch.tensor(self.sources, dtype=torch.float64)
        #self.targets = torch.tensor(self.targets, dtype=torch.float64)
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, index):
        return torch.FloatTensor(self.sources[index]), torch.FloatTensor(self.targets[index])

class DefenseTransformer(nn.Module):
    def __init__(self, input_dim = 23, output_dim =10, hidden_dim=128, num_layers=2, nhead=2):
        super(DefenseTransformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.outpur_fc = nn.Linear(output_dim, hidden_dim)

        self.transformer = nn.Transformer(d_model=hidden_dim, nhead = nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first= True)
        
        self.fc = nn.Linear(hidden_dim,output_dim)

    def forward(self, src, tgt):
        src = self.input_fc(src)
        tgt = self.outpur_fc(tgt)
        
        src_mask = self.transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        
        output = self.transformer(src, tgt, src_mask = src_mask, tgt_mask = tgt_mask)

        output = self.fc(output)
        return output

def train(model, DL, lr = 3e-5, epochs = 50, save_path = "./data/trained_model_with_mask.pth", plot_path = "./data/trained_loss_with_mask.png"):
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
        for samples, targets in DL:
            iter_count += 1
            if(iter_count == len(DL) -1 ): break
            samples, targets = samples.to(device), targets.to(device)

            # Forward pass
            output = model(samples, targets)

            # Compute the loss
            loss = criterion(output, targets)

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
    DS = BasketballDataset(filename)
    DL = DataLoader(DS, batch_size = 4, shuffle = True)
    model = DefenseTransformer()
    
    parser = argparse.ArgumentParser(description='Defense Transformer Model')
    parser.add_argument('--task', type=str, required=True, help='Specify the task, can be train or test')
    args = parser.parse_args()

    save_path = "./data/trained_model_with_mask_eighty.pth"
    
    

    if args.task:
        if args.task == "train":
            plot_path = "./data/trained_loss_with_mask_eighty.png"
            train(model = model, DL = DL, save_path= save_path, plot_path="./data/trained_loss_with_mask_eighty.png")
        elif args.task == "test":
            model.load_state_dict(torch.load(save_path))
            #for samples, targets in DL:
            #    CV.visualize_game(samples[0].numpy(), anim_path="./data/animation_game_real.gif", title = "Real Game")
            #    print("samples shape:", samples.shape)
            #    print("targets shape:", targets.shape)
            #    tmp = copy.deepcopy(samples.numpy())
            #    with torch.no_grad():
            #        for i in range(48):
            #            action = model(samples[:,0:i+1,:],targets[:,0:i+1,:])
            #            #print(tmp[:,i+1,13:23].shape)
            #            #print(action[:,i,:].shape)
            #            tmp[:,i+1,13:23] = tmp[:,i,13:23] + action[:,i,:].numpy()
            #            samples[:,i+1,13:23] = samples[:,i,13:23] + action[:,i,:]
            #        print("action shape:", action.shape)
            #    #tmp = tmp.numpy()
            #    CV.visualize_game(tmp[0], anim_path= f"./data/animation_game_transformer.gif", title= "Transformer Generated Game")


            test_source = torch.FloatTensor(test_source)
            test_target = torch.FloatTensor(test_target)

            samples = copy.deepcopy(test_source)
            targets = copy.deepcopy(test_target)
            CV.visualize_game(samples[0].numpy(), anim_path="./data/animation_game_real.gif", title = "Real Game")
            print("samples shape:", samples.shape)
            print("targets shape:", targets.shape)
            tmp = copy.deepcopy(samples.numpy())
            with torch.no_grad():
                for i in range(48):
                    action = model(samples[0:1,0:i+1,:],targets[0:1,0:i+1,:])
                    #print(tmp[:,i+1,13:23].shape)
                    #print(action[:,i,:].shape)
                    tmp[0:1,i+1,13:23] = tmp[0:1,i,13:23] + action[0:1,i,:].numpy()
                    samples[0:1,i+1,13:23] = samples[0:1,i,13:23] + action[0:1,i,:]
                print("action shape:", action.shape)
            #tmp = tmp.numpy()
            print(tmp - test_source.numpy())
            CV.visualize_game(tmp[0], anim_path= f"./data/animation_game_transformer.gif", title= "Transformer Generated Game")
            