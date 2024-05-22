import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from OffenseTransformer import *

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward_one(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return torch.from_numpy(self.trajectories[idx]).float()

if __name__ == "__main__":
    offense_dataset = OffensiveBasketballDataset("./data/50Real.npy")
    offense_dataloader = DataLoader(offense_dataset, batch_size=1, shuffle=True)

    test_source = torch.FloatTensor(offense_dataset.test_source)
    test_data = copy.deepcopy(test_source.numpy())
    train_source = torch.FloatTensor(offense_dataset.sources)
    train_data = copy.deepcopy(train_source.numpy())

    num_trajectories = train_data.shape[0]
    trajectory_length = 49
    trajectory_dims = 23

    input_dim = trajectory_length * trajectory_dims
    hidden_dim = 128
    embedding_dim = 64
    batch_size = 4
    num_epochs = 10

    train_dataset = TrajectoryDataset(train_data)
    test_dataset = TrajectoryDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = SiameseNetwork(input_dim, hidden_dim, embedding_dim)
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Training starts")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            input1, input2 = batch[0][:, :trajectory_dims], batch[0][:, :trajectory_dims]
            input1, input2 = input1.to(device), input2.to(device)
            optimizer.zero_grad()
            output1, output2 = model(input1.view(-1, input_dim), input2.view(-1, input_dim))
            #print(output1.shape, output2.shape)
            loss = criterion(output1, output2, torch.ones(input1.size(0)).to(device))  # Similarity label is 1
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

    torch.save(model.state_dict(), './data/trajectory_similarity_model.pth')