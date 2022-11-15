import random
from collections import deque, namedtuple
import numpy as np
import torch as torch
import torchvision as torchvision
from torchvision.transforms import InterpolationMode


class DDQN(torch.nn.Module):

    def __init__(self, state_dim, action_dim, lr=0.05):
        super().__init__()

        self.learning_rate = lr

        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(

            torch.nn.Conv2d(in_channels=state_dim,
                            out_channels=32,
                            kernel_size=8,
                            stride=4
                            ),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=4,
                            stride=2
                            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=64,
                            out_channels=64,
                            kernel_size=(4, 4)
                            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            torch.nn.Flatten(),

            torch.nn.Linear(
                in_features=576,
                out_features=576
            ),

            torch.nn.ReLU(),
            torch.nn.Linear(576, action_dim)
        )

    def forward(self, obs):
        return self.model(obs)


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# Class to handle agent memory
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, sample_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        batch = random.sample(self.memory, sample_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


# Agen class
class Steve:

    def __init__(self, learning_rate=0.001, gamma=0.99, tau=0.01, replay_buffer_size=10000):
        self.action_space = 8

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau

        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")

        self.model = DDQN(3, self.action_space)
        self.target_model = DDQN(3, self.action_space)

        self.memory = ReplayMemory(replay_buffer_size)

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def sel_action(self, state, eps=0.01):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        acts = self.model.forward(state)  # Choose best Action
        act = np.argmax(acts.detach().numpy())

        if random.random() > eps:
            act = random.randrange(0, self.action_space)

        return act

    def replay_memory(self, sample_size=20):
        if sample_size > len(self.memory):
            return

        states, actions, rewards, next_states, dones = self.memory.sample(sample_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)

        curr_Q = self.model.forward(states).gather(1, actions)
        next_Q = self.target_model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]  # TODO: Hva skjer her???
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)

        expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q

        # TODO: Vurder annen metode for loss-utregning
        loss = torch.nn.functional.mse_loss(curr_Q, expected_Q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def translate_action(self, action):
        forward = action == 0
        backward = action == 1
        left = action == 2
        right = action == 3
        cam_pitch_l = action == 4
        cam_pitch_r = action == 5
        cam_yaw_up = action == 6
        cam_yaw_down = action == 6

        return [
            1 if forward else 2 if backward else 0,
            1 if left else 2 if right else 0,
            0,
            11 if cam_pitch_l else 13 if cam_pitch_r else 12,
            11 if cam_yaw_up else 13 if cam_yaw_down else 12,
            0,
            0,
            0,
        ]

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_': self.optimizer.state_dict()
        }, path)