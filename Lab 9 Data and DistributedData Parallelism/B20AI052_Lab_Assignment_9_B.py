# Neccessary imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from PIL import Image
from datetime import datetime
from pathlib import Path


import ignite
import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.handlers import PiecewiseLinear
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger

import warnings
warnings.filterwarnings('ignore')

# Seting up the configuration
config = {
    "seed": 543,
    "data_path": "./data/",
    "output_path": "output/",
    "model": "resnet18",
    "batch_size1": 32,
    "batch_size2": 64,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "num_workers": 1,
    "num_epochs": 30,
    "learning_rate": 1e-3,
    "num_warmup_epochs": 1,
    "validate_every": 3,
    "checkpoint_every": 200,
    "backend": None,
    "resume_from": None,
    "log_every_iters": 15,
    "nproc_per_node": None,
    "with_clearml": False,
    "with_amp": False,
}

# Importing SVHN dataset
def get_train_test_datasets(path):
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.SVHN(path, split='train', download=True, transform=train_transform)
    test_dataset = datasets.SVHN(path, split='test', download=True, transform=test_transform)
    return train_dataset, test_dataset

def get_dataflow(config):
    train_dataset, test_dataset = get_train_test_datasets('./data')
    train_loader = idist.auto_dataloader(train_dataset, batch_size=config["batch_size2"], num_workers=config["num_workers"], shuffle=True)
    test_loader = idist.auto_dataloader(test_dataset, batch_size=2, num_workers=config["num_workers"], shuffle=False)
    return train_loader, test_loader

def get_model(config, num_classes=10):
    model_name = config["model"]
    if model_name in models.__dict__:
        fn = models.__dict__[model_name]
    else:
        raise RuntimeError(f"Unknown model name {model_name}")

    model = idist.auto_model(fn(num_classes=10))

    return model

def get_optimizer(config, model):
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"], nesterov=True)
    optimizer = idist.auto_optim(optimizer)
    return optimizer

def get_criterion():
    return nn.CrossEntropyLoss().to(idist.device())

def get_save_handler(config):
    if config["with_clearml"]:
        from ignite.contrib.handlers.clearml_logger import ClearMLSaver

        return ClearMLSaver(dirname=config["output_path"])

    return config["output_path"]

def load_checkpoint(resume_from):
    checkpoint_fp = Path(resume_from)
    assert (
        checkpoint_fp.exists()
    ), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
    checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
    return checkpoint


def create_trainer(model, optimizer, criterion, train_sampler, config, logger):

    device = idist.device()
    amp_mode = None
    scaler = False
        
    trainer = create_supervised_trainer(
        model,
        optimizer,
        criterion,
        device=device,
        non_blocking=True,
        output_transform=lambda x, y, y_pred, loss: {"batch loss": loss.item()},
        amp_mode="amp" if config["with_amp"] else None,
        scaler=config["with_amp"],
    )
    trainer.logger = logger

    to_save = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
    }
    metric_names = [
        "batch loss",
    ]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=config["checkpoint_every"],
        save_handler=get_save_handler(config),
        output_names=metric_names if config["log_every_iters"] > 0 else None,
        with_pbars=False,
        clear_cuda_cache=False,
    )

    if config["resume_from"] is not None:
        checkpoint = load_checkpoint(config["resume_from"])
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer

def create_evaluator(model, metrics, config):
    device = idist.device()

    amp_mode = "amp" if config["with_amp"] else None
    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device, non_blocking=True, amp_mode=amp_mode
    )
    return evaluator

def setup_rank_zero(logger, config):
    device = idist.device()

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = config["output_path"]
    folder_name = (
        f"{config['model']}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
    )

    output_path = Path(output_path) / folder_name
    if not output_path.exists():
        output_path.mkdir(parents=True)
    config["output_path"] = output_path.as_posix()
    logger.info(f"Output path: {config['output_path']}")

    if config["with_clearml"]:
        from clearml import Task

        task = Task.init("SVHN-Training", task_name=output_path.stem)
        task.connect_configuration(config)
        # Log hyper parameters
        hyper_params = [
            "model",
            "batch_size",
            "momentum",
            "weight_decay",
            "num_epochs",
            "learning_rate",
            "num_warmup_epochs",
        ]
        task.connect({k: v for k, v in config.items()})

def log_basic_info(logger, config):
    logger.info(f"Train on SVHN")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(
            f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}"
        )
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")

def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )

def training(local_rank, config):

    rank = idist.get_rank()

    logger = setup_logger(name="SVHN-Training")
    log_basic_info(logger, config)

    if rank == 0:
        setup_rank_zero(logger, config)

    train_loader, val_loader = get_dataflow(config)
    model = get_model(config)
    optimizer = get_optimizer(config, model)
    criterion = get_criterion()
    config["num_iters_per_epoch"] = len(train_loader)

    trainer = create_trainer(
        model, optimizer, criterion, train_loader.sampler, config, logger
    )

    metrics = {
        "Accuracy": Accuracy(),
        "Loss": Loss(criterion),
    }

    train_evaluator = create_evaluator(model, metrics, config)
    val_evaluator = create_evaluator(model, metrics, config)

    def run_validation(engine):
        epoch = trainer.state.epoch
        state = train_evaluator.run(train_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "train", state.metrics)
        state = val_evaluator.run(val_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "val", state.metrics)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config["validate_every"]) | Events.COMPLETED,
        run_validation,
    )

    if rank == 0:
        evaluators = {"train": train_evaluator, "val": val_evaluator}
        tb_logger = common.setup_tb_logging(
            config["output_path"], trainer, optimizer, evaluators=evaluators
        )

    best_model_handler = Checkpoint(
        {"model": model},
        get_save_handler(config),
        filename_prefix="best",
        n_saved=2,
        global_step_transform=global_step_from_engine(trainer),
        score_name="val_accuracy",
        score_function=Checkpoint.get_default_score_fn("Accuracy"),
    )
    val_evaluator.add_event_handler(
        Events.COMPLETED,
        best_model_handler,
    )

    try:
        trainer.run(train_loader, max_epochs=config["num_epochs"])
    except Exception as e:
        logger.exception("")
        raise e

    if rank == 0:
        tb_logger.close()


print("Results for 2 GPU with 2 processes")
spawn_kwargs = {}
spawn_kwargs["start_method"] = "fork"
spawn_kwargs["nproc_per_node"] = 2
spawn_kwargs["dist_url"] = "tcp://127.0.0.1:3333"
config["backend"] = "gloo"

with idist.Parallel(backend=config["backend"], **spawn_kwargs) as parallel:
    parallel.run(training, config)
print("Training completed for 2 GPUs with 2 processes")

