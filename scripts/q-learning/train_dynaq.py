import random
import time

import numpy as np
import torch

from utils import make_env
from dynaq import DynaQAgent
from trainer import Trainer
from logger import create_logger, TensorBoardLogger, WandbLogger, CompositeLogger
import argparse
import os
from distutils.util import strtobool
import yaml
from datetime import date, datetime

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


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    """
    Parse command-line arguments, overriding defaults from config.yaml.
    """
    defaults = load_config()
    default_exp_name = "exp_" + date.today().strftime("%Y%m%d") + "_" + datetime.now().strftime("%H%M%S")
    default_wandb_run_name = default_exp_name

    parser = argparse.ArgumentParser()
    
    # Experiment 
    parser.add_argument("--exp-name", type=str, default=default_exp_name)
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--total-timesteps", type=int, default=defaults["total_timesteps"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    
    # Logging (run-specific)
    parser.add_argument("--logger", type=str, default=defaults["logger"],
                        choices=["tensorboard", "wandb", "both", "console", "none"])
    parser.add_argument("--log-dir", type=str, default=defaults["log_dir"])
    parser.add_argument("--wandb-project-name", type=str, default=defaults["wandb_project_name"])
    parser.add_argument("--wandb-run-name", type=str, default=default_wandb_run_name)
    parser.add_argument("--wandb-entity", type=str, default=defaults["wandb_entity"])
    parser.add_argument("--record-period", type=int, default=defaults["record_period"])
    
    # Device and determinism
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=defaults["torch_deterministic"], nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=defaults["cuda"], nargs="?", const=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=defaults["track"], nargs="?", const=True)
    
    # Standard RL hyperparameters 
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--buffer-size", type=int, default=defaults["buffer_size"])
    parser.add_argument("--gamma", type=float, default=defaults["gamma"])
    parser.add_argument("--tau", type=float, default=defaults["tau"])
    parser.add_argument("--target-network-frequency", type=int, default=defaults["target_network_frequency"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--start-e", type=float, default=defaults["start_e"])
    parser.add_argument("--end-e", type=float, default=defaults["end_e"])
    parser.add_argument("--exploration-fraction", type=float, default=defaults["exploration_fraction"])
    parser.add_argument("--learning-starts", type=int, default=defaults["learning_starts"])
    parser.add_argument("--train-frequency", type=int, default=defaults["train_frequency"])
    
    # DynaQ-specific 
    parser.add_argument("--imagined-buffer-size", type=int, default=defaults["imagined_buffer_size"])
    parser.add_argument("--dream-switch-off-step", type=int, default=defaults["dream_switch_off_step"])
    parser.add_argument("--model-learning-rate", type=float, default=defaults["model_learning_rate"])
    parser.add_argument("--model-update-frequency", type=int, default=defaults["model_update_frequency"])
    parser.add_argument("--dream-rollout-length", type=int, default=defaults["dream_rollout_length"])
    parser.add_argument("--dream-frequency", type=int, default=defaults["dream_frequency"])
    parser.add_argument("--dream-batch-size", type=int, default=defaults["dream_batch_size"])
    parser.add_argument("--q-batch-ratio", type=float, default=defaults["q_batch_ratio"])

    args = parser.parse_args()
    return args

def setup_logger(args, run_name):
    """
    Create logger based on configuration.
    
    Args:
        args: Configuration arguments
        run_name: Name for this training run
    
    Returns:
        BaseLogger instance
    """
    logger_type = args.logger.lower()
    
    if logger_type == "tensorboard":
        return TensorBoardLogger(log_dir=f"{args.log_dir}/{run_name}")
    
    elif logger_type == "wandb":
        return WandbLogger(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args)
        )
    
    elif logger_type == "both":
        # Use both TensorBoard and W&B
        return CompositeLogger([
            TensorBoardLogger(log_dir=f"{args.log_dir}/{run_name}"),
            WandbLogger(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args)
            )
        ])
    
    elif logger_type == "console":
        from logger import ConsoleLogger
        return ConsoleLogger(verbose=True)
    
    elif logger_type == "none":
        from logger import NoOpLogger
        return NoOpLogger()
    
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")




args = parse_args()
run_name = args.wandb_run_name

logger = setup_logger(args, run_name)
logger.log_hyperparameters(vars(args))

# --- Seeding ---
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# --- Environment Setup ---
envs = make_env(args.env_id, args.seed, train=True, record_period=args.record_period)
obs, _ = envs.reset(seed=args.seed)
obs_shape = obs.shape
action_space = getattr(envs, "action_space", None)

# --- Agent Initialization ---
#agent = DynaQAgent(envs, args, device)
q_network = MLPQNetwork(obs_shape, envs.action_space.n).to(device)
q_optimizer = Adam(q_network.parameters(), lr=args.learning_rate)
target_network = MLPQNetwork(obs_shape, envs.action_space.n).to(device)

world_model = MLPWorldModel(obs_shape, envs.action_space.n).to(device)
model_optimizer = Adam(world_model.parameters(), lr=args.model_learning_rate)

# --- Debug Info ---
if action_space is not None and getattr(action_space, "shape", None) not in (None, ()):
    action_shape = action_space.shape
else:
    raise NotImplementedError("This code only supports environments with discrete action spaces.")
action_dim = envs.action_space.n
batch_shape = (args.batch_size,) + tuple(obs_shape)
print("[debug] Using device:", device)
print(f"[debug] observation shape: {obs_shape}")
print(f"[debug] action shape: {action_shape}")
print(f"[debug] batch tensor shape: {batch_shape}")
rb_real = ReplayBuffer(
    args.buffer_size,
    envs.observation_space,
    envs.action_space,
    device,
    handle_timeout_termination=False,
)

rb_imagined = ReplayBuffer(
    args.imagined_buffer_size,
    envs.observation_space,
    envs.action_space,
    device,
    handle_timeout_termination=False,
)
    

# --- Training ---
global_step = 0
latest_episodes_list = []
for episode in range(args.total_episodes):
    obs, _ = envs.reset(seed=args.seed)

    episode_transitions = []
    while True:
        # --- Select Action ---
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            args.total_episodes
        )
        if random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            # Unsqueeze to add batch dimension: (1, obs_dim)
            obs_tensor = torch.Tensor(obs).to(device).unsqueeze(0)
            # q_values shape: (1, action_dim)
            q_values = q_network(obs_tensor)
            actions = torch.argmax(q_values, dim=1).squeeze(-1).cpu().numpy()
        
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        global_step += 1

        real_done = terminations or truncations
        rb_real.add(obs, next_obs, actions, rewards, real_done, infos)

        # --- Dream Generation ---
        if global_step % args.dream_frequency == 0 and episode < args.dream_switch_off_episode:
            # Sample the most recent real experiences as dream starting points
            buf_size = rb_real.size()
            dream_batch_size = min(args.dream_batch_size, buf_size) 
            if rb_real.full:
                batch_indices = np.array([(i + rb_real.pos) % buf_size for i in range(dream_batch_size)])      
            else:
                batch_indices = np.arange(buf_size - dream_batch_size, buf_size)
            
            dream_starts = rb_real._get_samples(batch_indices)
            current_s = dream_starts.observations  # (dream_batch_size, obs_dim)
            
            for k in range(args.dream_rollout_length):
                with torch.no_grad():
                    # dream_q_values shape: (dream_batch_size, action_dim)
                    dream_q_values = q_network(current_s)
                    # dream_actions shape: (dream_batch_size,)
                    dream_actions = torch.argmax(dream_q_values, dim=1)
                    # dream_actions_one_hot shape: (dream_batch_size, action_dim)
                    dream_actions_one_hot = F.one_hot(dream_actions, action_dim).float()
                
                # (dream_batch_size, obs_dim), (dream_batch_size, 1), (dream_batch_size, 1)
                pred_s, pred_r, pred_d = world_model.dream_step(current_s, dream_actions_one_hot)
                
                for idx in range(args.dream_batch_size):
                    rb_imagined.add(
                        current_s[idx].detach().cpu().numpy(),
                        pred_s[idx].detach().cpu().numpy(),
                        dream_actions[idx].detach().cpu().numpy(),
                        pred_r[idx].detach().cpu().numpy(),
                        pred_d[idx].detach().cpu().numpy(),
                        [{}],
                    )
                
                # Recursive step: next dream state is the last predicted one
                current_s = pred_s 
        
        episode_transitions += [{
                "observations": obs,
                "actions": actions,
                "next_observations": next_obs,
                "rewards": rewards,
                "dones": real_done
            }]

        if episode < args.learning_starts_after_episode:
            continue

        # --- Update Q-Network ---
        total_q_loss = torch.tensor(0.0)
        total_q_loss_real = torch.tensor(0.0)
        total_q_loss_imagined = torch.tensor(0.0)
        for i in range(args.updates_per_steps):
            if global_step < args.dream_switch_off_step:
                real_batch_size = int(args.batch_size * args.q_batch_ratio)
            else:
                real_batch_size = args.batch_size
            # data_real shapes:
            #   observations: (real_batch_size, obs_dim)
            #   actions: (real_batch_size, 1)
            #   next_observations: (real_batch_size, obs_dim)
            #   rewards: (real_batch_size, 1)
            #   dones: (real_batch_size, 1)
            transitions_real = rb_real.sample(real_batch_size)
            
            imagined_batch_size = args.batch_size - real_batch_size
            transitions_imagined = rb_imagined.sample(imagined_batch_size)
            
            # real batches
            with torch.no_grad():
                # target_max shape: (real_batch_size,)
                target_max, _ = target_network(transitions_real.next_observations).max(dim=1)
                # td_target_real shape: (real_batch_size,)
                td_target_real = transitions_real.rewards.flatten() + args.gamma * target_max * (1 - transitions_real.dones.flatten())
            # old_val_real shape: (real_batch_size,)
            old_val_real = q_network(transitions_real.observations).gather(1, transitions_real.actions.long()).squeeze()
            loss_real = F.smooth_l1_loss(td_target_real, old_val_real)

            # imagined batches
            loss_imagined = torch.tensor(0.0).to(device)
            if global_step < args.dream_switch_off_step:
                with torch.no_grad():
                    # target_max shape: (imagined_batch_size,)
                    target_max, _ = target_network(transitions_imagined.next_observations).max(dim=1)
                    td_target_imagined = transitions_imagined.rewards.flatten() + args.gamma * target_max * (1 - transitions_imagined.dones.flatten())
                old_val_imagined = q_network(transitions_imagined.observations).gather(1, transitions_imagined.actions.long()).squeeze()
                loss_imagined = F.smooth_l1_loss(td_target_imagined, old_val_imagined)

            q_loss = loss_real + loss_imagined
            total_q_loss += q_loss
            total_q_loss_real += loss_real
            total_q_loss_imagined += loss_imagined

            q_optimizer.zero_grad()
            total_q_loss.backward()
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), 100.0)
            q_optimizer.step()


        # Soft update target network
        target_net_state_dict = target_network.state_dict()
        policy_net_state_dict = q_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*args.tau + target_net_state_dict[key]*(1-args.tau)
        target_network.load_state_dict(target_net_state_dict)

        # Log losses for this step
        if global_step % args.log_interval_steps == 0:
            logger.log_scalar("losses/q_loss_total", total_q_loss.item() / args.updates_per_steps, global_step)
            logger.log_scalar("losses/q_loss_real", total_q_loss_real.item() / args.updates_per_steps, global_step)
            logger.log_scalar("losses/q_loss_imagined", total_q_loss_imagined.item() / args.updates_per_steps, global_step)

        # Reset if done
        if real_done:
            break

    # --- Update World Model ---
    # Sample from real experience
    # Returns batch with shapes:
    #   observations: (batch_size, obs_dim)
    #   actions: (batch_size, 1)
    #   next_observations: (batch_size, obs_dim)
    #   rewards: (batch_size, 1)
    #   dones: (batch_size, 1)
    total_world_model_loss = torch.tensor(0.0)
    total_world_model_loss_state = torch.tensor(0.0)
    total_world_model_loss_reward = torch.tensor(0.0)
    total_world_model_loss_done = torch.tensor(0.0)

    if latest_episodes_list.__len__() >= args.model_update_episodes:
        latest_episodes_list.pop(0)
    latest_episodes_list.append(episode_transitions)
    
    total_transitions = 0
    for episode in range(len(latest_episodes_list)):
        episode_transitions = latest_episodes_list[episode]
        for transition in episode_transitions:
            with torch.no_grad():
                s_ = torch.from_numpy(transition.observations).to(device)              # (batch_size, obs_dim)
                # Convert actions to one-hot encoding: (batch_size, 1) -> (batch_size,)
                a_ = F.one_hot(transition.actions.long(), action_dim).float().squeeze(1)
                next_s_ = torch.from_numpy(transition.next_observations).to(device)    # (batch_size, obs_dim)
                r_ = torch.from_numpy(transition.rewards).to(device)                   # (batch_size, 1)
                d_ = torch.from_numpy(transition.dones).to(device)                     # (batch_size, 1)
            
            model_loss, l_s, l_r, l_d = world_model.get_model_loss(s_, a_, next_s_, r_, d_)
            model_optimizer.zero_grad()
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 100.0)
            model_optimizer.step()
        total_transitions += len(episode_transitions)
            
    # Log model losses
    logger.log_scalar("losses/total_world_model_loss", total_world_model_loss.item() / total_transitions, episode)
    logger.log_scalar("losses/model_loss_state", total_world_model_loss_state.item() / total_transitions, episode)
    logger.log_scalar("losses/model_loss_reward", total_world_model_loss_reward.item() / total_transitions, episode)
    logger.log_scalar("losses/model_loss_done", total_world_model_loss_done.item() / total_transitions, episode)

    # Log episode statistics
    logger.log_scalar("episode/reward", infos["episode"]["r"], episode)
    logger.log_scalar("episode/length", infos["episode"]["l"], episode)
    logger.log_scalar("epsilon", epsilon, episode)
    



print(f"Training complete! Run name: {run_name}")
final_model_path = os.path.join(".", "final_model.pt")
torch.save(q_network.state_dict(), final_model_path)
print(f"Saved final model to {final_model_path}")
