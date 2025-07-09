import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        
        self._init_weights()

    def _init_weights(self):
        for layer in [self.layer_1, self.layer_2, self.layer_3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        # Initialize final layer with smaller weights
        nn.init.uniform_(self.layer_3.weight, -3e-3, 3e-3)
        nn.init.constant_(self.layer_3.bias, 0)

    def forward(self, state):
        a = F.relu(self.layer_1(state))
        a = F.relu(self.layer_2(a))
        a = torch.tanh(self.layer_3(a)) * self.max_action
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        
        self._init_weights()

    def _init_weights(self):
        for layer in [self.layer_1, self.layer_2, self.layer_3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        q = F.relu(self.layer_1(state_action))
        q = F.relu(self.layer_2(q))
        q = self.layer_3(q)
        return q

class TD3:
    def __init__(self, state_dim, action_dim, max_action, device, learning_rate=3e-4):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=learning_rate)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=learning_rate)

        self.max_action = max_action
        self.device = device

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        for it in range(2):
            # Sample replay buffer
            state, action, next_state, reward, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)
            done = torch.FloatTensor(done).to(self.device)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (torch.randn_like(action, dtype=torch.float32) * policy_noise).clamp(-noise_clip, noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                target_Q1 = self.critic_1_target(next_state, next_action)
                target_Q2 = self.critic_2_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1 = self.critic_1(state, action)
            current_Q2 = self.critic_2(state, action)

            # Compute critic losses separately
            critic_1_loss = F.mse_loss(current_Q1, target_Q)
            critic_2_loss = F.mse_loss(current_Q2, target_Q)

            # Optimize the critics
            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
            self.critic_1_optimizer.step()

            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)
            self.critic_2_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_optimizer.step()

                # Update the frozen target networks
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic_1.state_dict(), filename + "_critic_1")
        torch.save(self.critic_2.state_dict(), filename + "_critic_2")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic_1.load_state_dict(torch.load(filename + "_critic_1"))
        self.critic_2.load_state_dict(torch.load(filename + "_critic_2"))

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = [], [], [], [], []
        
        for i in ind:
            state, action, next_state, reward, done = self.storage[i]
            batch_states.append(np.asarray(state, dtype=np.float32))
            batch_actions.append(np.asarray(action, dtype=np.float32))
            batch_next_states.append(np.asarray(next_state, dtype=np.float32))
            batch_rewards.append(np.asarray(reward, dtype=np.float32))
            batch_dones.append(np.asarray(done, dtype=np.float32))

        return (np.array(batch_states, dtype=np.float32), 
                np.array(batch_actions, dtype=np.float32), 
                np.array(batch_next_states, dtype=np.float32), 
                np.array(batch_rewards, dtype=np.float32).reshape(-1, 1), 
                np.array(batch_dones, dtype=np.float32).reshape(-1, 1))

def train_td3(max_episodes=200, max_steps=500, batch_size=100, discount=0.99, 
               tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, 
               exploration_noise=0.05, learning_rate=3e-4, model_name="td3_pendulum"):
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize TD3 with configurable learning rate
    td3 = TD3(state_dim, action_dim, max_action, device, learning_rate)
    replay_buffer = ReplayBuffer()

    writer = SummaryWriter(f'runs/{model_name}')

    total_rewards = []
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for _ in range(max_steps):
            action = td3.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=action_dim).astype(np.float32)
            action = action.clip(-max_action, max_action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.add((state, action, next_state, reward, done))

            if len(replay_buffer.storage) > batch_size:
                td3.train(replay_buffer, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

            state = next_state
            episode_reward += reward

            if done:
                break

        total_rewards.append(episode_reward)
        writer.add_scalar('Episode_Reward', episode_reward, episode)
        
        if episode % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
            writer.add_scalar('Average_Reward', avg_reward, episode)
        

    td3.save(f"models/{model_name}")
    writer.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(total_rewards)
    plt.title('TD3 Training Progress - Pendulum-v1')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.8)
    plt.savefig('static/td3_training_progress.png')

    return td3, total_rewards

if __name__ == "__main__":
    train_td3() 