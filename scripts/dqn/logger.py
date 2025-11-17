# Filename: logger.py
# Description: Flexible logging abstraction supporting multiple backends

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import os


class BaseLogger(ABC):
    """
    Abstract base class for loggers.
    Allows easy swapping between different loggers etc.
    """
    
    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int):
        pass
    
    @abstractmethod
    def log_text(self, tag: str, text: str, step: int = 0):
        pass
    
    @abstractmethod
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        pass
    
    @abstractmethod
    def close(self):
        pass


class TensorBoardLogger(BaseLogger):    
    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: Directory to save TensorBoard logs
        """
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
    
    def log_scalar(self, tag: str, value: float, step: int):
        """        
        Args:
            tag: Name of the metric (e.g., "losses/q_loss")
            value: Scalar value to log
            step: Global step/timestep
        """
        self.writer.add_scalar(tag, value, step)
    
    def log_text(self, tag: str, text: str, step: int = 0):
        """        
        Args:
            tag: Name/category of the text
            text: Text content to log
            step: Global step (default: 0)
        """
        self.writer.add_text(tag, text, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """
        Log hyperparameters as a table.
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        hparam_text = "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k, v in hparams.items()])
        self.log_text("hyperparameters", hparam_text)
    
    def close(self):
        self.writer.close()


class WandbLogger(BaseLogger):
    """Weights & Biases logging backend."""
    
    def __init__(self, project: str, entity: Optional[str] = None, 
                 name: Optional[str] = None, config: Optional[Dict] = None):
        """
        Args:
            project: W&B project name
            entity: W&B entity/team name (
            name: Run name 
            config: Configuration dictionary to log 
        """
        import wandb
        self.wandb = wandb
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            sync_tensorboard=False,
        )
    
    def log_scalar(self, tag: str, value: float, step: int):
        """
        Args:
            tag: Name of the metric (e.g., "losses/q_loss")
            value: Scalar value to log
            step: Global step/timestep
        """
        # Convert tag format from "category/name" to nested dict
        parts = tag.split('/')
        if len(parts) == 2:
            log_dict = {parts[0]: {parts[1]: value}}
        else:
            log_dict = {tag: value}
        
        self.wandb.log(log_dict, step=step)
    
    def log_text(self, tag: str, text: str, step: int = 0):
        """        
        Args:
            tag: Name/category of the text
            text: Text content to log
            step: Global step (default: 0)
        """
        self.wandb.log({tag: text}, step=step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """        
        Args:
            hparams: Dictionary of hyperparameters
        """
        self.wandb.config.update(hparams)
    
    def close(self):
        """Finish the W&B run."""
        self.wandb.finish()


class CompositeLogger(BaseLogger):
    """
    Composite logger that writes to multiple backends simultaneously.
    
    Example:
        logger = CompositeLogger([
            TensorBoardLogger("runs/experiment"),
            WandbLogger("my-project", name="experiment-1")
        ])
    """
    
    def __init__(self, loggers: list[BaseLogger]):
        """
        Args:
            loggers: List of logger instances to write to
        """
        self.loggers = loggers
    
    def log_scalar(self, tag: str, value: float, step: int):
        for logger in self.loggers:
            logger.log_scalar(tag, value, step)
    
    def log_text(self, tag: str, text: str, step: int = 0):
        for logger in self.loggers:
            logger.log_text(tag, text, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        for logger in self.loggers:
            logger.log_hyperparameters(hparams)
    
    def close(self):
        for logger in self.loggers:
            logger.close()


class ConsoleLogger(BaseLogger):
    """
    Terminal logger for debugging.
    Prints metrics to stdout instead of writing to files.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: If True, prints all metrics. If False, only prints summaries.
        """
        self.verbose = verbose
    
    def log_scalar(self, tag: str, value: float, step: int):
        if self.verbose:
            print(f"[Step {step}] {tag}: {value:.4f}")
    
    def log_text(self, tag: str, text: str, step: int = 0):
        if self.verbose:
            print(f"[{tag}]\n{text}\n")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        print("=" * 50)
        print("HYPERPARAMETERS")
        print("=" * 50)
        for key, value in hparams.items():
            print(f"{key:30s}: {value}")
        print("=" * 50)
    
    def close(self):
        pass


class NoOpLogger(BaseLogger):
    """
    No-operation logger that does nothing.
    Useful for disabling logging without changing code.
    """
    
    def log_scalar(self, tag: str, value: float, step: int):
        pass
    
    def log_text(self, tag: str, text: str, step: int = 0):
        pass
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        pass
    
    def close(self):
        pass


def create_logger(logger_type: str, **kwargs) -> BaseLogger:
    """
    This should be the only exposed function.
    
    Args:
        logger_type: Type of logger ("tensorboard", "wandb", "console", "none")
        **kwargs: Logger-specific arguments
    
    Returns:
        BaseLogger instance
    
    Examples:
        # TensorBoard
        logger = create_logger("tensorboard", log_dir="runs/exp1")
        
        # Weights & Biases
        logger = create_logger("wandb", project="my-project", name="exp1")
        
        # Console only
        logger = create_logger("console", verbose=True)
        
        # Multiple loggers
        logger = create_logger("composite", loggers=[
            create_logger("tensorboard", log_dir="runs/exp1"),
            create_logger("wandb", project="my-project")
        ])
    """
    logger_type = logger_type.lower()
    
    if logger_type == "tensorboard":
        return TensorBoardLogger(**kwargs)
    elif logger_type == "wandb":
        return WandbLogger(**kwargs)
    elif logger_type == "console":
        return ConsoleLogger(**kwargs)
    elif logger_type == "composite":
        return CompositeLogger(**kwargs)
    elif logger_type == "none":
        return NoOpLogger()
    else:
        raise ValueError(f"Unknown logger type: {logger_type}. "
                        f"Choose from: tensorboard, wandb, console, composite, none")
