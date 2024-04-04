import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np 

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        self.optimizer.zero_grad()
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        Q_pred = self.model(state)

        target = Q_pred.clone().detach()

        # Ensure action is a 1D tensor
        action = action.view(-1)

        # Convert action to numpy array
        action_numpy = action.numpy()

        # Compute target Q-value
        target[np.arange(len(target)), action_numpy] = reward + self.gamma * torch.max(self.model(next_state).detach(), dim=1)[0] * (1 - done)

        loss = self.criterion(Q_pred, target)
        loss.backward()

        self.optimizer.step()
