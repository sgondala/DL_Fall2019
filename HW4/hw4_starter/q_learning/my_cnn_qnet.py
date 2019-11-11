import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvQNet(nn.Module):
    def __init__(self, env, config, logger=None):
        super().__init__()

        #####################################################################
        # TODO: Define a CNN for the forward pass.
        #   Use the CNN architecture described in the following DeepMind
        #   paper by Mnih et. al.:
        #       https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        #
        # Some useful information:
        #     observation shape: env.observation_space.shape -> (H, W, C)
        #     number of actions: env.action_space.n
        #     number of stacked observations in state: config.state_history
        #####################################################################
        H, W, C = env.observation_space.shape
        self.first_layer = nn.Conv2d(C * config.state_history, 8, kernel_size = 4, stride = 2)
        self.relu1 = nn.ReLU()
        H_out = (H - (4-1) - 1) // 2 + 1
        self.second_layer = nn.Conv2d(8, 16, kernel_size = 4, stride = 2)
        self.relu2 = nn.ReLU()
        H_out = (H_out - (4-1) -1) // 2 + 1
        self.third_layer = nn.Conv2d(16, 32, kernel_size = 4, stride = 2)
        self.relu3 = nn.ReLU()
        H_out = (H_out - (4-1) -1) // 2 + 1
        self.linear_layer1 = nn.Linear(H_out * H_out * 32, H_out * H_out * 8)
        self.relu4 = nn.ReLU()
        self.linear_layer2 = nn.Linear(H_out * H_out * 8, env.action_space.n)
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################

    def forward(self, state):
        #####################################################################
        # TODO: Implement the forward pass.
        #####################################################################
        
        N = len(state)
        state = state.transpose(1,3) # N H W C => N C W H
        output = self.first_layer(state)
        output = self.relu1(output)
        output = self.second_layer(output)
        output = self.relu2(output)
        output = self.third_layer(output)
        output = self.relu3(output)
        output = output.reshape(N, -1)
        output = self.linear_layer1(output)
        output = self.relu4(output)
        output = self.linear_layer2(output)
        return output
        #####################################################################
        #                             END OF YOUR CODE                      #
        #####################################################################


# -8
        # H, W, C = env.observation_space.shape
        # self.first_layer = nn.Conv2d(C * config.state_history, 16, kernel_size = 8, stride = 4)
        # self.relu1 = nn.ReLU()
        # H_out = (H - (8-1) - 1) // 4 + 1
        # self.second_layer = nn.Conv2d(16, 32, kernel_size = 4, stride = 2)
        # self.relu2 = nn.ReLU()
        # H_out = (H_out - (4-1) -1) // 2 + 1
        # self.linear_layer1 = nn.Linear(H_out * H_out * 32, H_out * H_out * 8)
        # self.relu3 = nn.ReLU()
        # self.linear_layer2 = nn.Linear(H_out * H_out * 8, env.action_space.n)