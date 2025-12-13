import logging
from typing import Optional, Union

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from rich.logging import RichHandler

# from pytorch_lightning.utilities.rank_zero import rank_zero_only


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.addHandler(RichHandler())
    # file_formatter = logging.Formatter(fmt_file)
    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    # logging_levels = (
    #     "debug",
    #     "info",
    #     "warning",
    #     "error",
    #     "exception",
    #     "fatal",
    #     "critical",
    # )
    # for level in logging_levels:
    #     setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def get_wandb_logger(obj: Union[LightningModule, Trainer]) -> Optional[WandbLogger]:
    """Get the WandB logger from a LightningModule or Trainer, if available.
    
    Handles both single logger and multiple loggers (e.g., when using many_loggers).
    Returns None if no WandB logger is found.
    
    Args:
        obj: Either a LightningModule or Trainer instance
        
    Returns:
        The WandB logger if found, None otherwise
    """
    # Handle Trainer objects
    if isinstance(obj, Trainer):
        # First check if single logger is WandB
        if obj.logger is not None and isinstance(obj.logger, WandbLogger):
            return obj.logger
        # Then check all loggers (handles many_loggers case)
        if hasattr(obj, 'loggers'):
            for logger in obj.loggers:
                if isinstance(logger, WandbLogger):
                    return logger
        return None
    
    # Handle LightningModule objects
    if isinstance(obj, LightningModule):
        # First check if single logger is WandB
        if obj.logger is not None and isinstance(obj.logger, WandbLogger):
            return obj.logger
        # Then check trainer's loggers (handles many_loggers case)
        if hasattr(obj, 'trainer') and obj.trainer and hasattr(obj.trainer, 'loggers'):
            for logger in obj.trainer.loggers:
                if isinstance(logger, WandbLogger):
                    return logger
    
    return None
