from typing import NamedTuple

import torch._dynamo as dynamo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical


class Data(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        obs_shape = (4, 84, 84)
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_q = layer_init(nn.Linear(512, 16))

    def forward(self, x):
        x = F.relu(self.conv(x / 255.0))
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        obs_shape = obs_shape = (4, 84, 84)
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_logits = layer_init(nn.Linear(512, 16))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        logits = self(x / 255.0)
        policy_dist = Categorical(logits=logits)
        # GRAPH BREAK: Sampling from categorical distribution
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


def critic_update(actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, alpha, data, gamma):
    with torch.no_grad():
        _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
        qf1_next_target = qf1_target(data.next_observations)
        qf2_next_target = qf2_target(data.next_observations)
        # we can use the action probabilities instead of MC sampling to estimate the expectation
        min_qf_next_target = next_state_action_probs * (
            torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
        )
        # adapt Q-target for discrete Q-function
        min_qf_next_target = min_qf_next_target.sum(dim=1)
        next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * gamma * (min_qf_next_target)

        # use Q-values only for the taken actions
    qf1_values = qf1(data.observations)
    qf2_values = qf2(data.observations)
    qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
    qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
    qf_loss = qf1_loss + qf2_loss

    q_optimizer.zero_grad(True)
    qf_loss.backward()
    q_optimizer.step()
    return qf1_a_values, qf2_a_values, qf1_loss, qf2_loss, qf_loss


def actor_update(actor, qf1, qf2, actor_optimizer, alpha, data):
    _, log_pi, action_probs = actor.get_action(data.observations)
    with torch.no_grad():
        qf1_values = qf1(data.observations)
        qf2_values = qf2(data.observations)
        min_qf_values = torch.min(qf1_values, qf2_values)
        # no need for reparameterization, the expectation can be calculated for discrete actions
    actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

    actor_optimizer.zero_grad(True)
    actor_loss.backward()
    actor_optimizer.step()
    return log_pi, action_probs, actor_loss


def alpha_update(target_entropy, log_alpha, a_optimizer, log_pi, action_probs):
    alpha_loss = (action_probs.detach() * (-log_alpha * (log_pi + target_entropy).detach())).mean()

    a_optimizer.zero_grad(True)
    alpha_loss.backward()
    a_optimizer.step()
    # GRAPH BREAK: Returning alpha
    alpha = log_alpha.exp()
    return alpha, alpha_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor().to(device)
    qf1 = SoftQNetwork().to(device)
    qf2 = SoftQNetwork().to(device)
    qf1_target = SoftQNetwork().to(device)
    qf2_target = SoftQNetwork().to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=1e-4, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=1e-4, eps=1e-4)

    # Automatic entropy tuning
    target_entropy = -0.98 * torch.log(1 / torch.tensor(16))
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=1e-4, eps=1e-4)

    # Generate a fake batch of transitions
    obs = torch.randint(low=0, high=255, size=(32, 4, 84, 84), device=device)
    actions, log_pi, action_probs = actor.get_action(obs)
    data = Data(
        observations=obs,
        actions=actions.detach()[:, None],
        rewards=torch.randn(size=(32, 1), device=device),
        dones=torch.randint(low=0, high=2, size=(32, 1), device=device),
        next_observations=torch.randint(low=0, high=255, size=(32, 4, 84, 84), device=device),
    )

    breaks_critic = dynamo.explain(critic_update, actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, alpha, data, 0.99)[
        -1
    ]

    breaks_actor = dynamo.explain(actor_update, actor, qf1, qf2, actor_optimizer, alpha, data)[-1]

    breaks_alpha = dynamo.explain(alpha_update, target_entropy, log_alpha, a_optimizer, log_pi, action_probs)[-1]

    # import ipdb; ipdb.set_trace(context=21)

    # update the target networks
    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
        target_param.data.copy_(0.05 * param.data + (1 - 0.05) * target_param.data)
    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
        target_param.data.copy_(0.05 * param.data + (1 - 0.05) * target_param.data)