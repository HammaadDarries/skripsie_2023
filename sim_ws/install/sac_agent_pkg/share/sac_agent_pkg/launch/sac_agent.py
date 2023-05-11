import torch
import torch.nn as nn
import torch.nn.functional as F

NN_LAYER_1 = 100
NN_LAYER_2 = 100
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# MEMORY_SIZE = 100000
# SEED = 0
# TAU = 1e-2
# GAMMA = 0.99
# BATCH_SIZE = 100
# LR = 1e-3



class PolicyNetworkSAC(nn.Module):
    def __init__(self, num_inputs = 40, num_actions = 2):
        super(PolicyNetworkSAC, self).__init__()

        self.linear1 = nn.Linear(num_inputs, NN_LAYER_1)
        self.linear2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.mean_linear = nn.Linear(NN_LAYER_2, num_actions)
        self.log_std_linear = nn.Linear(NN_LAYER_2, num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        std = torch.exp(log_std)
        normal = torch.distributions.Normal(0, 1)

        z = mean + std * normal.sample().requires_grad_()
        action = torch.tanh(z)

        log_prob = torch.distributions.Normal(mean, std).log_prob(z) - torch.log(1 - action * action + EPSILON) 
        log_prob = log_prob.sum(-1, keepdim=True)
            
        return action, log_prob


class SACAgent:
    def __init__(self, model):
        self.actor = model

    def act(self, state):
        state = torch.FloatTensor(state)
        action, log_prob = self.actor(state)
        
        return action.detach().numpy()