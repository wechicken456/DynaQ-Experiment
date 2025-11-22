from abc import ABC
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def scale_reward_down(r):
    return (r - (-100.0)) / (100.0 - (-100.0))  # adjust according to environment reward range

def scale_reward_up(r):
    return r * (100.0 - (-100.0)) + (-100.0) # inverse of scale_reward_down

class MLPQNetwork(ABC, nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[64, 64]):
        """
        Args:
            obs_dim (int): Dimension of observation space
            action_dim (int): Number of discrete actions
            hidden_dims (list[int]): List of hidden layer dimensions
        """
        super().__init__()
        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass to compute Q-values.
        
        Args:
            x: Input state tensor
               Shape: (batch_size, obs_dim)
        
        Returns:
            Q-values for each action
            Shape: (batch_size, action_dim)
        """
        return self.network(x)



class MLPWorldModel(nn.Module):
    """
    The "Simulator" (CA3) component.
    A multi-head MLP network that learns to predict the environment's dynamics.
    It takes (state, action) and predicts (next_state, reward, done).
    Suitable for low-dimensional continuous state spaces.
    """
    def __init__(self, obs_dim, action_dim, hidden_dims=[64, 64], 
                 state_loss_weight=1.0, reward_loss_weight=1.0, done_loss_weight=1.0):
        """
        Args:
            obs_dim (int): Dimension of observation space
            action_dim (int): Number of discrete actions
            hidden_dims (list[int]): List of hidden layer dimensions
            state_loss_weight (float): Weight for state prediction loss
            reward_loss_weight (float): Weight for reward prediction loss
            done_loss_weight (float): Weight for done prediction loss
            use_layer_norm (bool): Whether to use layer normalization
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_loss_weight = state_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.done_loss_weight = done_loss_weight
        
        # Shared body for (state, action) features
        layers = []
        prev_dim = obs_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim
        self.shared_body = nn.Sequential(*layers)

        # Multi-head outputs
        # 1. Next-State Head (Predicts continuous state vector)
        self.head_state = nn.Linear(prev_dim, obs_dim)
        
        # 2. Reward Head (Predicts scalar reward)
        self.head_reward = nn.Linear(prev_dim, 1)
        
        # 3. Done Head (Predicts termination logit)
        self.head_done = nn.Linear(prev_dim, 1)

    def forward(self, s, a_one_hot) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns the delta between next imagined state and current state.
        Should call
        
        Args:
            s: Current state tensor
               Shape: (batch_size, obs_dim)
            a_one_hot: One-hot encoded action tensor
                       Shape: (batch_size, action_dim)
        
        Returns:
            tuple of (pred_next_state, pred_reward, pred_done_logit):
                pred_next_state: Predicted next state
                                Shape: (batch_size, obs_dim)
                pred_reward: Predicted reward
                            Shape: (batch_size, 1)
                pred_done_logit: Predicted termination logit
                                Shape: (batch_size, 1)
        """
        # sa shape: (batch_size, obs_dim + action_dim)
        sa = torch.cat([s, a_one_hot], dim=1)
        
        # features shape: (batch_size, hidden_dims[-1])
        features = self.shared_body(sa)
        
        # Predict next state, reward, and done
        pred_next_state = self.head_state(features)    # (batch_size, obs_dim)
        pred_reward = self.head_reward(features)        # (batch_size, 1)
        pred_done_logit = self.head_done(features)      # (batch_size, 1)
        
        return pred_next_state, pred_reward, pred_done_logit
    
    @torch.no_grad()
    def predict(self, s, a_one_hot) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        A single "dream" step. Predicts the next state, reward, and done.
        This is used for generative "imagination rollouts".
        
        Args:
            s: Current state tensor
               Shape: (batch_size, obs_dim)
            a_one_hot: One-hot encoded action tensor
                       Shape: (batch_size, action_dim)
        
        Returns:
            tuple of (pred_s, pred_r, pred_d):
                pred_s: Predicted next state
                       Shape: (batch_size, obs_dim)
                pred_r: Predicted reward
                       Shape: (batch_size, 1)
                pred_d: Predicted done flag (0 or 1)
                       Shape: (batch_size, 1)
        """
        # should still be same shapes
        delta_state, reward, done_logit = self.forward(s, a_one_hot)
        next_state = s + delta_state # Residual connection
        done_prob = torch.sigmoid(done_logit)
        return next_state, reward, done_prob

    def get_model_loss(self, s, a_one_hot, next_s, r, d):
        """
        Calculates the multi-part loss for training the World Model.
        
        Args:
            s: Current state tensor
               Shape: (batch_size, obs_dim)
            a_one_hot: One-hot encoded action tensor
                       Shape: (batch_size, action_dim)
            next_s: Target next state tensor
                    Shape: (batch_size, obs_dim)
            r: Target reward tensor
               Shape: (batch_size,) or (batch_size, 1)
            d: Target done flag tensor
               Shape: (batch_size,) or (batch_size, 1)
        
        Returns:
            tuple of (total_loss, loss_state, loss_reward, loss_done):
                total_loss: Weighted sum of all losses (scalar)
                loss_state: MSE loss for state prediction (scalar)
                loss_reward: MSE loss for reward prediction (scalar)
                loss_done: BCE loss for done prediction (scalar)
        """
        # shapes: (batch_size, obs_dim), (batch_size, 1), (batch_size, 1)
        pred_delta, pred_r, pred_d_logits = self.forward(s, a_one_hot)
        target_delta = next_s - s  # Residual target
        loss_state = F.smooth_l1_loss(pred_delta, target_delta)

        # Squeeze just to be sure
        loss_reward = F.smooth_l1_loss(pred_r.squeeze(), r.squeeze())
        
        if d.type() != torch.float32:
            d = d.float()
            
        loss_done = F.binary_cross_entropy_with_logits(pred_d_logits.squeeze(), d.squeeze())

        total_loss = (self.state_loss_weight * loss_state + 
                     self.reward_loss_weight * loss_reward + 
                     self.done_loss_weight * loss_done)
        return total_loss, loss_state, loss_reward, loss_done




#class CNNQNetwork(nn.Module):
#    """
#    CNN-based Q-Network for image-based environments (e.g., Atari).
#    Expects input shape (batch, channels, height, width).
#    """
#    def __init__(self, obs_shape, action_dim):
#        """
#        Args:
#            obs_shape (tuple): Shape of observation (C, H, W)
#                              e.g., (4, 84, 84) for Atari with frame stacking
#            action_dim (int): Number of discrete actions
#        """
#        super().__init__()
#        # Assume obs_shape is (C, H, W)
#        c, h, w = obs_shape
#        
#        self.conv = nn.Sequential(
#            # Input: (batch_size, C, H, W)
#            nn.Conv2d(c, 32, kernel_size=8, stride=4),
#            # Output: (batch_size, 32, H', W') where H' ≈ (H-8)/4 + 1
#            nn.ReLU(),
#            nn.Conv2d(32, 64, kernel_size=4, stride=2),
#            # Output: (batch_size, 64, H'', W'')
#            nn.ReLU(),
#            nn.Conv2d(64, 64, kernel_size=3, stride=1),
#            # Output: (batch_size, 64, H''', W''')
#            nn.ReLU(),
#            nn.Flatten(),
#            # Output: (batch_size, 64 * H''' * W''')
#        )
#        
#        # Calculate conv output size
#        with torch.no_grad():
#            dummy_input = torch.zeros(1, c, h, w)
#            conv_out_size = self.conv(dummy_input).shape[1]
#        
#        self.fc = nn.Sequential(
#            nn.Linear(conv_out_size, 512),
#            nn.ReLU(),
#            nn.Linear(512, action_dim),
#        )
#    
#    def forward(self, x):
#        """
#        Forward pass to compute Q-values from image observations.
#        
#        Args:
#            x: Input image tensor
#               Shape: (batch_size, C, H, W)
#        
#        Returns:
#            Q-values for each action
#            Shape: (batch_size, action_dim)
#        """
#        # Extract features through convolutional layers
#        # x shape: (batch_size, flattened_features)
#        x = self.conv(x)
#        # Compute Q-values through fully connected layers
#        # output shape: (batch_size, action_dim)
#        return self.fc(x)
