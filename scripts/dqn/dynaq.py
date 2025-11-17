# Filename: agent.py
# Description: DynaQ Agent implementation with integrated World Model

import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from cleanrl.cleanrl_utils.buffers import ReplayBuffer

from models import MLPQNetwork, MLPWorldModel
from utils import linear_schedule
import os
from argparse import Namespace

class DynaQAgent(nn.Module):
    """
    Dyna-Q Agent that combines:
    - Q-Learning (Selector/CA1)
    - World Model learning (Simulator/CA3)
    - Dream generation for planning
    """
    def __init__(self, envs, args, device, q_network_class=None, world_model_class=None):
        super().__init__()
        self.envs = envs
        self.args = args
        self.device = device
        
        # Environment dimensions
        self.obs_shape = np.array(envs.observation_space.shape).prod()
        self.action_dim = envs.action_space.n
        
        # Allow custom network architectures
        self.q_network_class = q_network_class or MLPQNetwork
        self.world_model_class = world_model_class or MLPWorldModel
        
        self._init_networks()
        self._init_buffers()
    
    def _build_state_dict(self):
        return {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "world_model": self.world_model.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "args": vars(self.args),
        }
    
    def save(self, path: str):
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self._build_state_dict(), path)
    
    @classmethod
    def load(cls, path: str, env, device):
        """
        Load agent state from a checkpoint.
        
        Args:
            path: Path to the checkpoint file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=device)
        args = Namespace(**checkpoint["args"])
        agent = cls(env, args, device)

        agent.q_network.load_state_dict(checkpoint["q_network"])
        agent.target_network.load_state_dict(checkpoint["target_network"])
        agent.world_model.load_state_dict(checkpoint["world_model"])
        agent.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        agent.model_optimizer.load_state_dict(checkpoint["model_optimizer"])
        return agent
        
    def _init_networks(self):
        """Initialize Q-Network, Target Network, and World Model."""
        self.q_network = self.q_network_class(self.obs_shape, self.action_dim).to(self.device)
        self.q_optimizer = Adam(self.q_network.parameters(), lr=self.args.learning_rate)
        self.target_network = self.q_network_class(self.obs_shape, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.world_model = self.world_model_class(self.obs_shape, self.action_dim).to(self.device)
        self.model_optimizer = Adam(self.world_model.parameters(), lr=self.args.model_learning_rate)
    
    def _init_buffers(self):
        """Initialize replay buffers for real and imagined experiences."""
        # Stores experience from the actual environment
        self.rb_real = ReplayBuffer(
            self.args.buffer_size,
            self.envs.observation_space,
            self.envs.action_space,
            self.device,
            handle_timeout_termination=False,
        )

        self.rb_imagined = ReplayBuffer(
            self.args.imagined_buffer_size,
            self.envs.observation_space,
            self.envs.action_space,
            self.device,
            handle_timeout_termination=False,
        )
    
    def select_action(self, obs, global_step):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            obs: Observation from environment
                 Shape: (obs_dim,) - single observation
            global_step: Current training step for epsilon scheduling
        
        Returns:
            tuple of (actions, epsilon):
                actions: Selected action (scalar or array)
                epsilon: Current exploration rate (float)
        """
        epsilon = linear_schedule(
            self.args.start_e, 
            self.args.end_e, 
            self.args.exploration_fraction * self.args.total_timesteps, 
            global_step
        )
        
        if random.random() < epsilon:
            actions = self.envs.action_space.sample()
        else:
            # Unsqueeze to add batch dimension: (1, obs_dim)
            obs_tensor = torch.Tensor(obs).to(self.device).unsqueeze(0)
            # q_values shape: (1, action_dim)
            q_values = self.q_network(obs_tensor)
            actions = torch.argmax(q_values, dim=1).squeeze(-1).cpu().numpy()
        
        return actions, epsilon

    def store_real_experience(self, obs : np.ndarray, next_obs : np.ndarray, actions : np.ndarray, rewards : np.ndarray, done : np.ndarray, infos : list):
        """Store real experience in the real buffer."""
        self.rb_real.add(obs, next_obs, actions, rewards, done, infos)

    def train_world_model(self, global_step, logger):
        """
        "SIMULATION"
        Train the World Model to predict environment dynamics.
        
        Args:
            global_step: Current training step
            logger: Logger instance for metrics
        """
        if global_step % self.args.model_update_frequency != 0 or global_step >= self.args.dream_switch_off_step:
            return
        
        # Sample from real experience
        # Returns batch with shapes:
        #   observations: (batch_size, obs_dim)
        #   actions: (batch_size, 1)
        #   next_observations: (batch_size, obs_dim)
        #   rewards: (batch_size, 1)
        #   dones: (batch_size, 1)
        real_data = self.rb_real.sample(self.args.batch_size)
        
        with torch.no_grad():
            s_ = real_data.observations              # (batch_size, obs_dim)
            # Convert actions to one-hot encoding: (batch_size, 1) -> (batch_size,)
            a_ = F.one_hot(real_data.actions.long(), self.action_dim).float().squeeze(1)
            next_s_ = real_data.next_observations    # (batch_size, obs_dim)
            r_ = real_data.rewards                   # (batch_size, 1)
            d_ = real_data.dones                     # (batch_size, 1)
        
        model_loss, l_s, l_r, l_d = self.world_model.get_model_loss(s_, a_, next_s_, r_, d_)
        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()
        
        # Log losses
        if global_step % (self.args.model_update_frequency * 10) == 0:
            logger.log_scalar("losses/world_model_loss", model_loss.item(), global_step)
            logger.log_scalar("losses/model_loss_state", l_s.item(), global_step)
            logger.log_scalar("losses/model_loss_reward", l_r.item(), global_step)
            logger.log_scalar("losses/model_loss_done", l_d.item(), global_step)
    
    def generate_dreams(self, global_step):
        """
        "DREAMING"
        Generate imagined experiences using the World Model.
        
        Args:
            global_step: Current training step
        """
        if global_step % self.args.dream_frequency != 0 or global_step >= self.args.dream_switch_off_step:
            return
        
        # Sample the most recent real experiences as dream starting points
        buf_size = self.rb_real.size()
        dream_batch_size = min(self.args.dream_batch_size, buf_size) 
        if self.rb_real.full:
            batch_indices = np.array([(i + self.rb_real.pos) % buf_size for i in range(dream_batch_size)])      
        else:
            batch_indices = np.arange(buf_size - dream_batch_size, buf_size)
        
        dream_starts = self.rb_real._get_samples(batch_indices)
        current_s = dream_starts.observations  # (dream_batch_size, obs_dim)
        
        for k in range(self.args.dream_rollout_length):
            with torch.no_grad():
                # dream_q_values shape: (dream_batch_size, action_dim)
                dream_q_values = self.q_network(current_s)
                # dream_actions shape: (dream_batch_size,)
                dream_actions = torch.argmax(dream_q_values, dim=1)
                # dream_actions_one_hot shape: (dream_batch_size, action_dim)
                dream_actions_one_hot = F.one_hot(dream_actions, self.action_dim).float()
            
            # (dream_batch_size, obs_dim), (dream_batch_size, 1), (dream_batch_size, 1)
            pred_s, pred_r, pred_d = self.world_model.dream_step(current_s, dream_actions_one_hot)
            
            for idx in range(self.args.dream_batch_size):
                self.rb_imagined.add(
                    current_s[idx].detach().cpu().numpy(),
                    pred_s[idx].detach().cpu().numpy(),
                    dream_actions[idx].detach().cpu().numpy(),
                    pred_r[idx].detach().cpu().numpy(),
                    pred_d[idx].detach().cpu().numpy(),
                    [{}],
                )
            
            # Recursive step: next dream state is the last predicted one
            current_s = pred_s  
    
    def train_q_network(self, global_step, logger):
        """
        Train Q-Network on both real and imagined experiences.
        
        Args:
            global_step: Current training step
            logger: Logger instance for metrics
        """
        if global_step % self.args.train_frequency != 0:
            return
        
        if global_step < self.args.dream_switch_off_step:
            real_batch_size = int(self.args.batch_size * self.args.q_batch_ratio)
        else:
            real_batch_size = self.args.batch_size
        # data_real shapes:
        #   observations: (real_batch_size, obs_dim)
        #   actions: (real_batch_size, 1)
        #   next_observations: (real_batch_size, obs_dim)
        #   rewards: (real_batch_size, 1)
        #   dones: (real_batch_size, 1)
        data_real = self.rb_real.sample(real_batch_size)
        
        imagined_batch_size = self.args.batch_size - real_batch_size
        data_imagined = self.rb_imagined.sample(imagined_batch_size)
        
        # Train on real batches
        with torch.no_grad():
            # target_max shape: (real_batch_size,)
            target_max, _ = self.target_network(data_real.next_observations).max(dim=1)
            # td_target_real shape: (real_batch_size,)
            td_target_real = data_real.rewards.flatten() + self.args.gamma * target_max * (1 - data_real.dones.flatten())
        # old_val_real shape: (real_batch_size,)
        old_val_real = self.q_network(data_real.observations).gather(1, data_real.actions.long()).squeeze()
        loss_real = F.mse_loss(td_target_real, old_val_real)

        # Train on imagined batches
        loss_imagined = torch.tensor(0.0).to(self.device)
        if global_step < self.args.dream_switch_off_step:
            with torch.no_grad():
                # target_max shape: (imagined_batch_size,)
                target_max, _ = self.target_network(data_imagined.next_observations).max(dim=1)
                td_target_imagined = data_imagined.rewards.flatten() + self.args.gamma * target_max * (1 - data_imagined.dones.flatten())
            old_val_imagined = self.q_network(data_imagined.observations).gather(1, data_imagined.actions.long()).squeeze()
            loss_imagined = F.mse_loss(td_target_imagined, old_val_imagined)

        total_q_loss = loss_real + loss_imagined

        self.q_optimizer.zero_grad()
        total_q_loss.backward()
        self.q_optimizer.step()

        # Log losses
        if global_step % (self.args.train_frequency * 10) == 0:
            logger.log_scalar("losses/q_loss_total", total_q_loss.item(), global_step)
            logger.log_scalar("losses/q_loss_real", loss_real.item(), global_step)
            logger.log_scalar("losses/q_loss_imagined", loss_imagined.item(), global_step)
    
    def update_target_network(self, global_step):
        """Update target network weights."""
        if global_step % self.args.target_network_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
