import random
import time

import numpy as np
import torch

from utils import make_env
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
from cleanrl_utils.buffers import ReplayBuffer

from models import MLPQNetwork, MLPWorldModel, scale_reward_down
from utils import linear_schedule
import os
import wandb
from collections import deque

def load_config(config_path="config_dqn.yaml"):
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
    parser.add_argument("--total-episodes", type=int, default=defaults["total_episodes"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--stop-reward", type=float, default=320.0)
    
    # Logging (run-specific)
    parser.add_argument("--logger", type=str, default=defaults["logger"],
                        choices=["tensorboard", "wandb", "both", "console", "none"])
    parser.add_argument("--log-dir", type=str, default=defaults["log_dir"])
    parser.add_argument("--wandb-project-name", type=str, default=defaults["wandb_project_name"])
    parser.add_argument("--wandb-run-name", type=str, default=defaults["wandb_run_name"])
    parser.add_argument("--wandb-entity", type=str, default=defaults["wandb_entity"])
    parser.add_argument("--record-period", type=int, default=defaults["record_period"])
    parser.add_argument("--log-interval-steps", type=int, default=defaults["log_interval_steps"])

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
    parser.add_argument("--epsilon-start", type=float, default=defaults["epsilon_start"])
    parser.add_argument("--epsilon-end", type=float, default=defaults["epsilon_end"])
    parser.add_argument("--epsilon-decay", type=float, default=defaults["epsilon_decay"])
    parser.add_argument("--learning-starts-after-episode", type=int, default=defaults["learning_starts_after_episode"])
    parser.add_argument("--train-frequency", type=int, default=defaults["train_frequency"])
    parser.add_argument("--updates-per-step", type=int, default=defaults["updates_per_step"])

    
    # DynaQ-specific 
    parser.add_argument("--imagined-buffer-size", type=int, default=defaults["imagined_buffer_size"])
    parser.add_argument("--dream-start-episode", type=int, default=defaults["dream_start_episode"])
    parser.add_argument("--dream-switch-off-episode", type=int, default=defaults["dream_switch_off_episode"])
    parser.add_argument("--model-learning-rate", type=float, default=defaults["model_learning_rate"])
    parser.add_argument("--avg-episode-length", type=int, default=defaults["avg_episode_length"])
    parser.add_argument("--model-update-episodes", type=int, default=defaults["model_update_episodes"])
    parser.add_argument("--dream-rollout-length", type=int, default=defaults["dream_rollout_length"])
    parser.add_argument("--dream-frequency", type=int, default=defaults["dream_frequency"])
    parser.add_argument("--dream-batch-size", type=int, default=defaults["dream_batch_size"])
    parser.add_argument("--q-batch-ratio", type=float, default=defaults["q_batch_ratio"])

    args = parser.parse_args()
    return args




args = parse_args()
run_name = args.wandb_run_name

run = wandb.init(
    project=args.wandb_project_name,
    entity=args.wandb_entity,
    name=run_name,
    config=vars(args),
    sync_tensorboard=False,
)

wandb.config.update(vars(args))
run.define_metric(name="q_losses/*", step_metric="global_step")
run.define_metric(name="model_losses/*", step_metric="episode")
run.define_metric(name="episode/*", step_metric="episode")


# --- Seeding ---
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# --- Environment Setup ---
envs = make_env(args.env_id, args.seed, run_name, train=True, record_period=args.record_period)
obs, _ = envs.reset(seed=args.seed)
obs_shape = obs.shape # tuple
action_space = getattr(envs, "action_space", None)

# --- Agent Initialization ---
#agent = DynaQAgent(envs, args, device)
print(f"obs_shape: {obs_shape}, action_space.n: {envs.action_space.n}")
q_network = MLPQNetwork(obs_shape[0], envs.action_space.n).to(device)
q_optimizer = Adam(q_network.parameters(), lr=args.learning_rate)
target_network = MLPQNetwork(obs_shape[0], envs.action_space.n).to(device)
target_network.load_state_dict(q_network.state_dict())  # Initialize target network

world_model = MLPWorldModel(obs_shape[0], envs.action_space.n).to(device)
model_optimizer = Adam(world_model.parameters(), lr=args.model_learning_rate)

# --- Debug Info ---
if action_space is not None and getattr(action_space, "n", None).size > 0:
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

rb_dream_starts = ReplayBuffer(
    args.model_update_episodes * args.avg_episode_length,
    envs.observation_space,
    envs.action_space,
    device,
    handle_timeout_termination=False,
)

# --- Training ---
global_step = 0
latest_episodes_list = []
num_dream_starts = 0 # also the total number of transitions stored in latest_episodes_list above

epsilon = args.epsilon_start
reward_history = deque(maxlen=100)  # Store the last 100 episode rewards
best_reward = -float("inf")
for episode in range(args.total_episodes):
    obs, _ = envs.reset()

    episode_transitions = {
        "observations": [],
        "actions": [],
        "next_observations": [],
        "rewards": [],
        "dones": []
    }
    while True:
        # --- Select Action ---
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
        
        # Store transitions for world model training
        episode_transitions["observations"].append(obs)
        episode_transitions["actions"].append(actions)
        episode_transitions["next_observations"].append(next_obs)
        episode_transitions["rewards"].append(rewards)
        episode_transitions["dones"].append(real_done)
        num_dream_starts += 1
        
        # Update observation for next step
        obs = next_obs
        
        if real_done:
            break

        if episode < args.learning_starts_after_episode:
            continue

        # --- Dream Generation ---
        if global_step % args.dream_frequency == 0 and episode >= args.dream_start_episode and episode < args.dream_switch_off_episode:
            # Sample the most recent real experiences as dream starting points
            buf_size = rb_real.size()
            batch_size = min(args.dream_frequency, buf_size)
            if rb_dream_starts.size() >= batch_size:
                dream_starts = rb_dream_starts.sample(batch_size)
                current_s = dream_starts.observations.to(device)  # (batch_size, obs_dim)
                current_done = dream_starts.dones.to(device)      # (batch_size, 1)

                for k in range(args.dream_rollout_length):
                    with torch.no_grad():
                        # dream_q_values shape: (batch_size, action_dim)
                        dream_q_values = q_network(current_s)
                        # dream_actions shape: (batch_size,)
                        dream_actions = torch.argmax(dream_q_values, dim=1)
                        # dream_actions_one_hot shape: (batch_size, action_dim)
                        dream_actions_one_hot = F.one_hot(dream_actions, action_dim).float()
                    
                    # (batch_size, obs_dim), (batch_size, 1), (batch_size, 1)
                    pred_s, pred_r, pred_d = world_model.predict(current_s, dream_actions_one_hot)

                    for idx in range(batch_size):
                        if not current_done[idx]:   # don't dream further if already done
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
        
        # --- Update Q-Network ---
        if global_step % args.train_frequency == 0:
            total_q_loss = 0.0
            # total_q_loss_real = 0.0
            # total_q_loss_imagined = 0.0
            for i in range(args.updates_per_step):
                if episode < args.dream_switch_off_episode:
                    real_batch_size = int(args.batch_size * args.q_batch_ratio)
                    if rb_imagined.size() < args.batch_size - real_batch_size:  # make sure we have enough imagined samples
                        real_batch_size = args.batch_size - rb_imagined.size()
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
                
                # # real batches
                # with torch.no_grad():
                #     # target_max shape: (real_batch_size,)
                #     target_max, _ = target_network(transitions_real.next_observations).max(dim=1)
                #     # td_target_real shape: (real_batch_size,)
                #     td_target_real = transitions_real.rewards.flatten() + args.gamma * target_max * (1 - transitions_real.dones.flatten())
                # # main_val_real shape: (real_batch_size,)
                # main_val_real = q_network(transitions_real.observations).gather(1, transitions_real.actions.long()).squeeze()
                # loss_real = F.smooth_l1_loss(td_target_real, main_val_real)

                # # imagined batches
                # loss_imagined = torch.tensor(0.0).to(device)
                # if episode < args.dream_switch_off_episode and imagined_batch_size > 0:
                #     with torch.no_grad():
                #         # target_max shape: (imagined_batch_size,)
                #         target_max, _ = target_network(transitions_imagined.next_observations).max(dim=1)
                #         td_target_imagined = transitions_imagined.rewards.flatten() + args.gamma * target_max * (1 - transitions_imagined.dones.flatten())
                #     main_val_imagined = q_network(transitions_imagined.observations).gather(1, transitions_imagined.actions.long()).squeeze()
                #     loss_imagined = F.smooth_l1_loss(td_target_imagined, main_val_imagined)
                if imagined_batch_size > 0:
                    batch_states = torch.cat([transitions_real.observations, transitions_imagined.observations], dim=0)
                    batch_actions = torch.cat([transitions_real.actions, transitions_imagined.actions], dim=0)
                    batch_next_states = torch.cat([transitions_real.next_observations, transitions_imagined.next_observations], dim=0)
                    batch_rewards = torch.cat([transitions_real.rewards, transitions_imagined.rewards], dim=0)
                    batch_dones = torch.cat([transitions_real.dones, transitions_imagined.dones], dim=0)
                else:
                    batch_states = transitions_real.observations
                    batch_actions = transitions_real.actions
                    batch_next_states = transitions_real.next_observations
                    batch_rewards = transitions_real.rewards
                    batch_dones = transitions_real.dones

                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                batch_next_states = batch_next_states.to(device)
                batch_rewards = batch_rewards.to(device)
                batch_dones = batch_dones.to(device)
                
                with torch.no_grad():
                    target_q, _ = target_network(batch_next_states).max(dim=1)
                    td_target = batch_rewards.flatten() + args.gamma * target_q * (1 - batch_dones.flatten())
                current_q = q_network(batch_states).gather(1, batch_actions.long()).squeeze()
                q_loss = F.smooth_l1_loss(td_target, current_q)

                # q_loss = loss_real + loss_imagined
                total_q_loss += q_loss.detach().cpu().item()
                # total_q_loss_real += loss_real.detach().cpu().item()
                # total_q_loss_imagined += loss_imagined.detach().cpu().item()

                q_optimizer.zero_grad()
                q_loss.backward()
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
                # wandb.log({"losses/q_loss_total": total_q_loss / args.updates_per_step, "global_step": global_step})
                # wandb.log({"losses/q_loss_real": total_q_loss_real / args.updates_per_step, "global_step": global_step})
                # wandb.log({"losses/q_loss_imagined": total_q_loss_imagined / args.updates_per_step, "global_step": global_step})
                wandb.log({"losses/total_q_loss": total_q_loss / args.updates_per_step, "global_step": global_step})

        

    # Log episode statistics
    wandb.log({"episode/reward": infos["episode"]["r"], "episode": episode})
    wandb.log({"episode/length": infos["episode"]["l"], "episode": episode})
    wandb.log({"epsilon": epsilon, "episode": episode})
    epsilon = max(args.epsilon_decay * epsilon, args.epsilon_end)
    reward_history.append(infos["episode"]["r"])
    if episode % 100 == 0:
        running_avg = np.mean(reward_history)
        print(f"Episode {episode}, running avg reward : {running_avg}")
        if running_avg > best_reward:
            best_reward = running_avg
            print(f"New best reward: {best_reward}. Saving this model...")
            torch.save(q_network.state_dict(), f"best_model_{run_name}.pt")
        if running_avg >= args.stop_reward:
            print(f"Solved environment in {episode} episodes!")
            break

    if episode >= args.dream_switch_off_episode:
        continue

    # --- Update World Model ---
    # Sample from real experience
    # Returns batch with shapes:
    #   observations: (batch_size, obs_dim)
    #   actions: (batch_size, 1)
    #   next_observations: (batch_size, obs_dim)
    #   rewards: (batch_size, 1)
    #   dones: (batch_size, 1)

    latest_episodes_list.append(episode_transitions)
    if num_dream_starts >= args.model_update_episodes * args.avg_episode_length:
        total_world_model_loss = 0.0
        total_world_model_loss_delta = 0.0
        total_world_model_loss_reward = 0.0
        total_world_model_loss_done = 0.0
        
        total_transitions = 0
        rb_dream_starts.reset()
        rng = np.random.default_rng()  
        for ep_idx in range(len(latest_episodes_list)):
            episode_transitions = latest_episodes_list[ep_idx]
            obs = np.array(episode_transitions["observations"])
            actions = np.array(episode_transitions["actions"])
            next_obs = np.array(episode_transitions["next_observations"])
            rewards = np.array(episode_transitions["rewards"])
            dones = np.array(episode_transitions["dones"])
            rng.shuffle(obs)
            rng.shuffle(actions)
            rng.shuffle(next_obs)
            rng.shuffle(rewards)
            rng.shuffle(dones)

            rb_dream_starts.extend(obs, next_obs, actions, rewards, dones, [{}]*len(obs))

            for i in range(0, len(obs), args.dream_batch_size):
                end_idx = min(i + args.dream_batch_size, len(obs))
                with torch.no_grad():
                    s_ = torch.from_numpy(obs[i:end_idx]).to(device)              # (batch_size, obs_dim)
                    # Convert actions to one-hot encoding: (batch_size, 1) -> (batch_size,)
                    a_ = F.one_hot(torch.from_numpy(actions[i:end_idx]), action_dim).float().to(device)  # (batch_size, action_dim)
                    next_s_ = torch.from_numpy(next_obs[i:end_idx]).to(device)    # (batch_size, obs_dim)
                    r_ = torch.from_numpy(rewards[i:end_idx]).to(device)           # (batch_size, 1)
                    d_ = torch.from_numpy(dones[i:end_idx]).to(device)             # (batch_size, 1)
                
                total_model_loss, l_s, l_r, l_d = world_model.get_model_loss(s_, a_, next_s_, r_, d_)
                model_optimizer.zero_grad()
                total_model_loss.backward()
                torch.nn.utils.clip_grad_norm_(world_model.parameters(), 100.0)
                model_optimizer.step()
                total_world_model_loss += total_model_loss.detach().cpu().item()
                total_world_model_loss_delta += l_s.detach().cpu().item()
                total_world_model_loss_reward+= l_r.detach().cpu().item()
                total_world_model_loss_done += l_d.detach().cpu().item()
            total_transitions += len(episode_transitions)
                
        # Log model losses
        wandb.log({"model_losses/total": total_world_model_loss / total_transitions, "episode": episode})
        wandb.log({"model_losses/state": total_world_model_loss_delta / total_transitions, "episode": episode})
        wandb.log({"model_losses/reward": total_world_model_loss_reward / total_transitions, "episode": episode})
        wandb.log({"model_losses/done": total_world_model_loss_done / total_transitions, "episode": episode})

        num_dream_starts = 0
        latest_episodes_list = []



print(f"Training complete! Run name: {run_name}")
final_model_path = os.path.join(".", f"final_model_{run_name}.pt")
torch.save(q_network.state_dict(), final_model_path)
print(f"Saved final model to {final_model_path}")
