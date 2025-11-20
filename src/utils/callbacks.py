"""Callbacks for pytorch lightning trainer."""

import os
import platform
from typing import TYPE_CHECKING

import numpy as np
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import utils
from utils.transforms import UnNormalize

if TYPE_CHECKING:
    from pytorch_lightning import Trainer

log = utils.get_pylogger(__name__)


def _ensure_wandb_media_directory(trainer: "Trainer") -> None:
    """Ensure wandb media directory exists to prevent FileNotFoundError on Windows.
    
    This is a workaround for a known issue on Windows with wandb offline mode where
    directories aren't created before wandb tries to move temp files to them.
    Only applies on Windows to avoid potential issues on other platforms.
    
    Args:
        trainer: PyTorch Lightning Trainer instance
    
    Raises:
        RuntimeError: If the directory cannot be created or determined
        OSError: If directory creation fails due to filesystem errors
    """
    # Only apply this workaround on Windows
    if platform.system() != 'Windows':
        return  # On non-Windows, assume directory handling works
    
    # Handle case where no logger is configured
    if not trainer.logger:
        raise RuntimeError("Cannot ensure wandb media directory: no logger configured")
    
    loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
    for logger in loggers:
        # Only handle WandbLogger to avoid affecting other loggers
        if isinstance(logger, WandbLogger):
            wandb_dir = None
            # Try to get wandb directory from logger
            if hasattr(logger, 'experiment') and logger.experiment and hasattr(logger.experiment, 'dir'):
                try:
                    wandb_dir = logger.experiment.dir
                except (AttributeError, RuntimeError) as e:
                    log.debug(f"Could not get wandb dir from logger.experiment: {e}")
            
            # Alternative: check wandb.run.dir if available
            if not wandb_dir and hasattr(wandb, 'run') and wandb.run:
                try:
                    if hasattr(wandb.run, 'dir'):
                        wandb_dir = wandb.run.dir
                except (AttributeError, RuntimeError) as e:
                    log.debug(f"Could not get wandb dir from wandb.run: {e}")
            
            # Create directory structure if we found a valid wandb directory
            if wandb_dir and isinstance(wandb_dir, str) and wandb_dir.strip():
                # Normalize the path to handle any issues
                wandb_dir = os.path.normpath(wandb_dir)
                try:
                    # First ensure the wandb run directory itself exists
                    os.makedirs(wandb_dir, exist_ok=True)
                    # Then create all parent directories for media
                    os.makedirs(os.path.join(wandb_dir, 'files'), exist_ok=True)
                    media_base = os.path.join(wandb_dir, 'files', 'media')
                    os.makedirs(media_base, exist_ok=True)
                    # Create both images and table directories
                    images_dir = os.path.join(media_base, 'images')
                    table_dir = os.path.join(media_base, 'table')
                    os.makedirs(images_dir, exist_ok=True)
                    os.makedirs(table_dir, exist_ok=True)
                    # Also create common namespace subdirectories that wandb might use
                    # (e.g., 'test/' from 'test/acc_per_class' namespace)
                    # Normalize path to handle Windows separator issues
                    table_test_dir = os.path.normpath(os.path.join(table_dir, 'test'))
                    os.makedirs(table_test_dir, exist_ok=True)
                    # Verify the directories actually exist
                    if not os.path.isdir(images_dir):
                        raise RuntimeError(f"wandb images directory {images_dir} was not created successfully")
                    if not os.path.isdir(table_dir):
                        raise RuntimeError(f"wandb table directory {table_dir} was not created successfully")
                    if not os.path.isdir(table_test_dir):
                        raise RuntimeError(f"wandb table test directory {table_test_dir} was not created successfully")
                    return  # Successfully created or already exists
                except (OSError, PermissionError) as e:
                    raise OSError(f"Failed to create wandb media directory structure (base: {wandb_dir}): {e}") from e
            
            # If we found a WandbLogger but couldn't get the directory, raise an error
            raise RuntimeError("Could not determine wandb directory for media files")
    
    # If no WandbLogger was found
    raise RuntimeError("Cannot ensure wandb media directory: no WandbLogger found")


def ensure_wandb_test_logging_ready(pl_module, module_log=None) -> bool:
    """Shared guard logic for modules that log wandb tables at test time."""
    logger = module_log or log

    if not hasattr(pl_module, "trainer") or pl_module.trainer is None:
        logger.warning("Trainer not available, skipping wandb table logging")
        return False

    if not hasattr(pl_module, "logger") or pl_module.logger is None:
        logger.warning("Logger not available, skipping wandb table logging")
        return False

    if not hasattr(pl_module, "targets_test_all") or pl_module.targets_test_all is None:
        logger.warning("Test targets not available, skipping wandb table logging")
        return False

    if not hasattr(pl_module, "preds_test_all") or pl_module.preds_test_all is None:
        logger.warning("Test predictions not available, skipping wandb table logging")
        return False

    try:
        _ensure_wandb_media_directory(pl_module.trainer)
    except (RuntimeError, OSError) as e:
        logger.warning(f"Failed to ensure wandb media directories: {e}. Skipping table logging.")
        return False

    return True


class FreezeAllButLast(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        for name, param in pl_module.net.named_parameters():
            param.requires_grad = False

        # TODO: Make generic or add model name to know how to select last layers
        # TODO: Make generic or add model name to know how to select last layers
        # For most models. Ohne classification layer named classifier
        if hasattr(pl_module.net, "classifier"):
            pl_module.net.classifier.weight.requires_grad = True
            pl_module.net.classifier.bias.requires_grad = True
        # Deit: Distillation and CLS classifier
        if hasattr(pl_module.net, "cls_classifier"):
            pl_module.net.cls_classifier.weight.requires_grad = True
            pl_module.net.cls_classifier.bias.requires_grad = True
        # Deit: Distillation and CLS classifier
        if hasattr(pl_module.net, "cls_classifier"):
            pl_module.net.distillation_classifier.weight.requires_grad = True
            pl_module.net.distillation_classifier.bias.requires_grad = True

        for name, param in pl_module.net.named_parameters():
            log.debug(f"{name}: requires_grad={param.requires_grad}")

    def finetune_function(self, pl_module, epoch, optimizer) -> None:
        pass


class LogPredictionSamplesCallback(Callback):
    def __init__(self, n: int = 4):
        super().__init__()
        self.n = n

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        if batch_idx == 0:
            _ensure_wandb_media_directory(trainer)
            
            try:
                x, y = batch
                images = [img for img in x[: self.n]]
                idx2label = trainer.datamodule.idx2label
                captions = [
                    f"gt: {idx2label[y_i.item()]} | pred: {idx2label[pred_i.item()]}"
                    for y_i, pred_i in zip(y[: self.n], outputs["preds"][: self.n])
                ]
                trainer.logger.experiment.log(
                    {"prediction_samples": [wandb.Image(img, caption=cap) for img, cap in zip(images, captions)]},
                    commit=False,
                )
            except (FileNotFoundError, OSError) as e:
                log.warning(f"Failed to log prediction samples: {e}")
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)


class LogTrainingSamplesCallback(Callback):
    def __init__(self, n: int = 4):
        super().__init__()
        self.n = n

    def on_train_start(self, trainer, pl_module) -> None:
        dm = trainer.datamodule
        # original_images = [dm.train[i][0] for i in range(0, self.n)]
        # labels = [dm.idx2label[dm.train[i][1]] for i in range(0, self.n)]
        loader = DataLoader(dataset=dm.train, batch_size=self.n, num_workers=0, shuffle=True)
        samples = next(iter(loader))
        labels = [dm.idx2label[label_i.item()] for label_i in samples[1]]

        _ensure_wandb_media_directory(trainer)
        
        try:
            trainer.logger.experiment.log(
                {
                    "transformed_training_samples": [
                        wandb.Image(img, caption=cap) for img, cap in zip(list(samples[0]), labels)
                    ]
                },
                commit=False,
            )
        except (FileNotFoundError, OSError) as e:
            log.warning(f"Failed to log training samples: {e}")
        return super().on_train_start(trainer, pl_module)


class LogTrainingSamplesMultiDataParallelLoaderCallback(Callback):
    def __init__(self, n: int = 4):
        super().__init__()
        self.n = n

    def on_train_start(self, trainer, pl_module) -> None:
        dm = trainer.datamodule
        # TODO: Probably needs fix after MultiConcatDataLoader is implemented
        
        _ensure_wandb_media_directory(trainer)
        
        try:
            loader = DataLoader(dataset=dm.train_src[0], batch_size=self.n, num_workers=0, shuffle=True)
            samples = next(iter(loader))
            labels = [dm.idx2label[label_i.item()] for label_i in samples[1]]
            trainer.logger.experiment.log(
                {
                    "transformed_training_samples_source": [
                        wandb.Image(img, caption=cap) for img, cap in zip(list(samples[0]), labels)
                    ]
                },
                commit=False,
            )

            loader = DataLoader(dataset=dm.train_target[0], batch_size=self.n, num_workers=0, shuffle=True)
            samples = next(iter(loader))
            labels = [dm.idx2label[label_i.item()] for label_i in samples[1]]
            trainer.logger.experiment.log(
                {
                    "transformed_training_samples_target": [
                        wandb.Image(img, caption=cap) for img, cap in zip(list(samples[0]), labels)
                    ]
                },
                commit=False,
            )
        except (FileNotFoundError, OSError) as e:
            log.warning(f"Failed to log training samples: {e}")
        return super().on_train_start(trainer, pl_module)


class LogLayersRequiresGrad(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module) -> None:
        for name, param in pl_module.net.named_parameters():
            log.info(f"{name}: requires_grad={param.requires_grad}")
        return super().on_train_start(trainer, pl_module)
