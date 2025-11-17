# Filename: trainer.py
# Description: Reusable training loop for RL agents

import time
from logger import BaseLogger

class Trainer:
    """
    Generic RL training loop that can work with different agents and environments.
    Handles common training concerns like logging, environment resets, and episode tracking.
    """
    def __init__(self, agent, envs, args, logger: BaseLogger = None, print_debug_shapes=True):
        """
        Args:
            agent: RL agent instance (must implement select_action, store_real_experience, etc.)
            envs: Gymnasium environment instance
            args: Configuration/arguments object
            logger: Logger instance (BaseLogger) for metrics tracking
        """
        self.agent = agent
        self.envs = envs
        self.args = args
        self.logger = logger
        self.start_time = time.time()
        if print_debug_shapes:
            self._print_debug_shapes()

    def _print_debug_shapes(self):
        obs, _ = self.envs.reset(seed=self.args.seed)
        obs_shape = obs.shape
        action_space = getattr(self.envs, "action_space", None)
        if action_space is not None and getattr(action_space, "shape", None) not in (None, ()):
            action_shape = action_space.shape
        elif hasattr(self.agent, "action_dim"):
            action_shape = (self.agent.action_dim,)
        else:
            action_shape = ("unknown",)
        batch_shape = (self.args.batch_size,) + tuple(obs_shape)
        print("[debug] Using device:", self.agent.device)
        print(f"[debug] observation shape: {obs_shape}")
        print(f"[debug] action shape: {action_shape}")
        print(f"[debug] batch tensor shape: {batch_shape}")
        
    def train(self):
        """Main training loop."""
        obs, _ = self.envs.reset(seed=self.args.seed)
        
        for global_step in range(self.args.total_timesteps):
            obs, info = self._step_environment(obs, global_step)
            
            if global_step < self.args.learning_starts:
                continue
            
            self._training_step(global_step)
            
            # Periodic logging
            if global_step % (self.args.train_frequency * 10) == 0:
                self._log_metrics(global_step)
        
        self._cleanup()
    
    def _step_environment(self, obs, global_step):
        """
        Execute one step in the environment.
        
        Args:
            obs: Current observation
                 Shape: (obs_dim,)
            global_step: Current training step
        
        Returns:
            tuple of (next_obs, infos):
                next_obs: Next observation (shape: (obs_dim,))
                infos: Environment info dict
        """
        actions, epsilon = self.agent.select_action(obs, global_step)
        next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
        
        # Log episode statistics
        if "episode" in infos:
            self._log_episode_info(infos["episode"], global_step, epsilon)
        
        # Store experience
        real_done = terminations or truncations
        self.agent.store_real_experience(obs, next_obs, actions, rewards, real_done, infos)
        
        # Reset if done
        if real_done:
            next_obs, _ = self.envs.reset(seed=self.args.seed)
        
        return next_obs, infos
    
    def _training_step(self, global_step):
        """
        Execute one training step (can be overridden for different algorithms).
        
        Args:
            global_step: Current training step
        """
        raise NotImplemented
        #self.agent.train_world_model(global_step, self.logger)
        #self.agent.generate_dreams(global_step)
        #self.agent.train_q_network(global_step, self.logger)
        #self.agent.update_target_network(global_step)
    
    def _log_episode_info(self, info, global_step, epsilon):
        """
        Log 1 single episode statistics.
        
        Args:
            info: episode data
            global_step: Current training step
            epsilon: Current exploration rate
        """
        print(f"global_step={global_step}, episodic_return={info['r']}")
        if self.logger:
            self.logger.log_scalar("episode/reward", info["r"], global_step)
            self.logger.log_scalar("episode/length", info["l"], global_step)
            self.logger.log_scalar("epsilon", epsilon, global_step)
                
    
    def _log_metrics(self, global_step):
        """
        Args:
            global_step: Current training step
        """
        if self.logger:
            sps = int(global_step / (time.time() - self.start_time))
            self.logger.log_scalar("charts/SPS", sps, global_step)
    
    def _cleanup(self):
        self.envs.close()
        if self.logger:
            self.logger.close()


