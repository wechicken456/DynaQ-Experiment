# Filename: utils.py
# Description: Utility functions for Dyna-Q algorithm

import gymnasium as gym


def make_env(env_id, seed, train=True, render_mode="rgb_array", record_period=100):
    """
    Creates and wraps the LunarLander environment.
    """
    if render_mode:
        env = gym.make(env_id, render_mode=render_mode)
    else:
        env = gym.make(env_id)
    
    if render_mode == "rgb_array":
        env = gym.wrappers.RecordVideo(
            env, 
            video_folder="./videos",
            name_prefix="training" if train else "eval",
            episode_trigger=lambda x: x % record_period == 0
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Calculates the epsilon for epsilon-greedy exploration."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
