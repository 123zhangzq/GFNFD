import torch
import torch.nn as nn
import torch.nn.functional as F

class GFlowNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(GFlowNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return F.softmax(self.fc2(x), dim=-1)

    def select_action(self, state):
        probs = self.forward(state)
        return torch.multinomial(probs, 1).squeeze()
