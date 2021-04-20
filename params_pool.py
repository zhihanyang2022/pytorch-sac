import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal, Independent

from replay_buffer import Batch

def get_net(
        num_in:int,
        num_out:int,
        final_activation,  # e.g. nn.Tanh
        num_hidden_layers:int=1,
        num_neurons_per_hidden_layer:int=64
    ) -> nn.Sequential:

    layers = []

    layers.extend([
        nn.Linear(num_in, num_neurons_per_hidden_layer),
        nn.ReLU(),
    ])

    for _ in range(num_hidden_layers):
        layers.extend([
            nn.Linear(num_neurons_per_hidden_layer, num_neurons_per_hidden_layer),
            nn.ReLU(),
        ])

    layers.append(nn.Linear(num_neurons_per_hidden_layer, num_out))

    if final_activation is not None:
        layers.append(final_activation)

    return nn.Sequential(*layers)

class NormalPolicyNet(nn.Module):

    """Outputs a distribution with parameters learnable by gradient descent."""

    def __init__(self, input_dim, action_dim):
        super(NormalPolicyNet, self).__init__()
        self.shared_net = get_net(num_in=input_dim, num_out=64, final_activation=nn.ReLU())
        self.means_net  = get_net(num_in=64, num_out=action_dim, final_activation=None)
        self.stds_net   = get_net(num_in=64, num_out=action_dim, final_activation=nn.Softplus())

    def forward(self, states: torch.tensor):
        out = self.shared_net(states)
        means, stds = self.means_net(out), self.stds_net(out)
        return Independent(Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)

class QNet(nn.Module):

    """Has little quirks; just a wrapper so that I don't need to call concat many times"""

    def __init__(self, input_dim, action_dim):
        super(QNet, self).__init__()
        self.net = get_net(num_in=input_dim+action_dim, num_out=1, final_activation=None)

    def forward(self, states: torch.tensor, actions: torch.tensor):
        return self.net(torch.cat([states, actions], dim=1))

class ParamsPool:

    def __init__(self, input_dim, action_dim):

        self.Normal   = NormalPolicyNet(input_dim=input_dim, action_dim=action_dim)
        self.Normal_optimizer = optim.Adam(self.Normal.parameters(), lr=1e-3)

        self.Q1       = QNet(input_dim=input_dim, action_dim=action_dim)
        self.Q1_targ  = QNet(input_dim=input_dim, action_dim=action_dim)
        self.Q1_targ.load_state_dict(self.Q1.state_dict())
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=1e-3)

        self.Q2       = QNet(input_dim=input_dim, action_dim=action_dim)
        self.Q2_targ  = QNet(input_dim=input_dim, action_dim=action_dim)
        self.Q2_targ.load_state_dict(self.Q2.state_dict())
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=1e-3)

        self.gamma = 0.95
        self.alpha = 0.1
        self.polyak = 0.95

    # ==================================================================================================================
    # Helper methods (it is generally not my style of using helper methods but here they improve readability)
    # ==================================================================================================================

    def min_i_12(self, a: torch.tensor, b: torch.tensor) -> torch.tensor:
        return torch.minimum(a, b)

    def sample_action_and_compute_log_pi(self, state: torch.tensor, use_reparametrization_trick: bool) -> tuple:
        mu_given_s = self.Normal(state)  # in paper, mu represents the normal distribution
        # in paper, u represents the un-squashed action; nu stands for next u's
        # actually, we can just use reparametrization trick in both Step 12 and 14, but it might be good to separate
        # the two cases for pedagogical purposes, i.e., using reparametrization trick is a must in Step 14
        u = mu_given_s.rsample() if use_reparametrization_trick else mu_given_s.sample()
        a = torch.tanh(u)
        log_pi_a_given_s = mu_given_s.log_prob(u) - torch.sum(torch.log(1 - torch.tanh(u) ** 2), dim=1)
        return a, log_pi_a_given_s

    def clip_gradient(self, net: nn.Module) -> None:
        for param in net.parameters():
            param.grad.data.clamp_(-1, 1)

    def polyak_update(self, old_net: nn.Module, new_net: nn.Module) -> None:
        for old_param, new_param in zip(old_net.parameters(), new_net.parameters()):
            old_param.data.copy_(old_param.data * self.polyak + new_param.data * (1 - self.polyak))

    # ==================================================================================================================
    # Methods for learning
    # ==================================================================================================================

    def update_networks(self, b: Batch) -> None:

        # ========================================
        # Step 12: calculating targets
        # ========================================

        with torch.no_grad():

            na, log_pi_na_given_ns = self.sample_action_and_compute_log_pi(b.ns, use_reparametrization_trick=False)
            targets = b.r + self.gamma * (1 - b.d) * \
                      (self.min_i_12(self.Q1_targ(b.ns, na), self.Q2_targ(b.ns, na)) - self.alpha * log_pi_na_given_ns)

        # ========================================
        # Step 13: learning the Q functions
        # ========================================

        Q1_predictions = self.Q1(b.s, b.a)
        Q1_loss = torch.mean((Q1_predictions - targets) ** 2)

        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.clip_gradient(net=self.Q1)
        self.Q1_optimizer.step()

        Q2_predictions = self.Q2(b.s, b.a)
        Q2_loss = torch.mean((Q2_predictions - targets) ** 2)

        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.clip_gradient(net=self.Q2)
        self.Q2_optimizer.step()

        # ========================================
        # Step 14: learning the policy
        # ========================================

        a, log_pi_a_given_s = self.sample_action_and_compute_log_pi(b.s, use_reparametrization_trick=True)
        policy_loss = - torch.mean(self.min_i_12(self.Q1(b.s, a), self.Q2(b.s, a)) - self.alpha * log_pi_a_given_s)

        self.Normal_optimizer.zero_grad()
        policy_loss.backward()
        self.clip_gradient(net=self.Normal)
        self.Normal_optimizer.step()

        # ========================================
        # Step 15: update target networks
        # ========================================

        self.polyak_update(old_net=self.Q1_targ, new_net=self.Q1)
        self.polyak_update(old_net=self.Q2_targ, new_net=self.Q2)

    def act(self, state: np.array) -> np.array:
        state = torch.tensor(state).unsqueeze(0).float()
        action, _ = self.sample_action_and_compute_log_pi(state, use_reparametrization_trick=False)
        return action.numpy()[0]  # no need to detach first because we are not using the reparametrization trick






