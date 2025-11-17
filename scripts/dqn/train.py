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

class DynaQTrainer(Trainer):
    def _training_step(self, global_step):
        self.agent.train_world_model(global_step, self.logger)
        self.agent.generate_dreams(global_step)
        self.agent.train_q_network(global_step, self.logger)
        self.agent.update_target_network(global_step)

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

if __name__ == "__main__":
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
    
    # --- Agent Initialization ---
    agent = DynaQAgent(envs, args, device)
    
    # --- Training ---
    trainer = DynaQTrainer(agent, envs, args, logger)
    trainer.train()
    
    print(f"Training complete! Run name: {run_name}")
    final_model_path = os.path.join(".", "final_model.pt")
    agent.save(final_model_path)
    print(f"Saved final model to {final_model_path}")
