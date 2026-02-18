import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PPOActor(nn.Module):
    """
    Policy Network: Input latent z, Output mean delta_z and std.
    Action is modification to latent vector.
    """
    def __init__(self, latent_dim=256):
        super(PPOActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(256, latent_dim)
        self.log_std = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, z):
        h = self.net(z)
        mu = self.mu_head(h)
        std = torch.exp(self.log_std)
        return mu, std

class PPOCritic(nn.Module):
    """
    Value Network: Input latent z, Output scalar Value.
    """
    def __init__(self, latent_dim=256):
        super(PPOCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, z):
        return self.net(z)

class PPOOptimizer:
    def __init__(self, latent_dim=256, lr=3e-4, gamma=0.99, eps_clip=0.2, device='cpu'):
        self.actor = PPOActor(latent_dim).to(device)
        self.critic = PPOCritic(latent_dim).to(device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.device = device
        self.mse = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            mu, std = self.actor(state)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(dim=-1)
        return action, action_logprob

    def update(self, states, actions, logprobs, rewards, next_states, dones):
        """
        PPO Update Step.
        states: (B, 256)
        actions: (B, 256)
        logprobs: (B,)
        rewards: (B,)
        """
        # Convert to tensor
        states = states.to(self.device).detach()
        actions = actions.to(self.device).detach()
        old_logprobs = logprobs.to(self.device).detach()
        rewards = rewards.to(self.device).detach()

        # Monte Carlo estimate of returns (Simplified: 1-step or N-step?)
        # For Molecular Design, usually 1-step Optimization: State -> Action -> New State -> Reward.
        # So Return = Reward. (Gamma=0 for 1-step).
        # Let's assume immediate reward only for this version.
        returns = rewards

        # Normalize rewards
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Multiple PPO epochs
        for _ in range(4):
            # Evaluate Old Actions
            mu, std = self.actor(states)
            dist = torch.distributions.Normal(mu, std)
            logprobs = dist.log_prob(actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)

            state_values = self.critic(states).squeeze()

            # Ratios
            ratios = torch.exp(logprobs - old_logprobs)

            # Surrogate Loss
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss_actor = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()
            loss_critic = self.mse(state_values, returns)

            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()

        print("PPO Update Complete.")

# Mock Reward Function for Testing
def reward_function_mock(z):
    # Reward = magnitude of z (dummy)
    return torch.norm(z, dim=1)
