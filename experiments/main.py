import logging
from sacred import Experiment
import numpy as np
import seml

import sys
import warnings

sys.path.append(".")

from models import GeneralModel
from models.statistics.Metrics import Metrics
from utils.config_utils import *
from utils.model_utils import *
from utils.system_utils import *

import torch

from copy import deepcopy

ex = Experiment()
seml.setup_logger(ex)

def main(arguments,metrics:Metrics):
    global out
    out = metrics.log_line
    out(f"starting at {get_date_stamp()}")

    # get device
    device = configure_device(arguments)

    if arguments['disable_cuda_benchmark']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    

    # set seed 
    configure_seeds(arguments, device)


    # get model
    model: GeneralModel = find_right_model(
        NETWORKS_DIR, arguments['model'],
        device=device,
        hidden_dim=arguments['hidden_dim'],
        input_dim=arguments['input_dim'],
        output_dim=arguments['output_dim'],
        is_maskable=arguments['disable_masking'],
        is_tracking_weights=arguments['track_weights'],
        is_rewindable=arguments['enable_rewinding'],
        is_growable=arguments['growing_rate'] > 0,
        outer_layer_pruning=arguments['outer_layer_pruning'],
        maintain_outer_mask_anyway=(
                                       not arguments['outer_layer_pruning']) and (
                                           "Structured" in arguments['prune_criterion']),
        l0=arguments['l0'],
        l0_reg=arguments['l0_reg'],
        N=arguments['N'],
        beta_ema=arguments['beta_ema'],
        l2_reg=arguments['l2_reg']
    ).to(device)

    # get criterion
    criterion = find_right_model(
        CRITERION_DIR, arguments['prune_criterion'],
        model=model,
        limit=arguments['pruning_limit'],
        start=arguments['lower_limit'],
        steps=arguments['snip_steps'],
        device=arguments['device'],
        arguments=arguments,
        lower_limit=arguments['lower_limit']
    )

    # load data
    train_loader, test_loader = find_right_model(
        DATASETS, arguments['data_set'],
        arguments=arguments,
        mean=arguments['mean'],
        std=arguments['std']
    )

    # get loss function
    loss = find_right_model(
        LOSS_DIR, arguments['loss'],
        device=device,
        l1_reg=arguments['l1_reg'],
        lp_reg=arguments['lp_reg'],
        l0_reg=arguments['l0_reg'],
        hoyer_reg=arguments['hoyer_reg']
    )

    # get optimizer
    optimizer = find_right_model(
        OPTIMS, arguments['optimizer'],
        params=model.parameters(),
        lr=arguments['learning_rate'],
        # momentum=arguments['momentum'],
        weight_decay=arguments['l2_reg'] if not arguments['l0'] else 0
    )

    run_name = f'_model={arguments["model"]}_dataset={arguments["data_set"]}_prune-criterion={arguments["prune_criterion"]}' + \
               f'_pruning-limit={arguments["pruning_limit"]}_train-scheme={arguments["train_scheme"]}_seed={arguments["seed"]}'
    
    # build trainer
    trainer = find_right_model(
        TRAINERS_DIR, arguments['train_scheme'],
        model=model,
        loss=loss,
        optimizer=optimizer,
        device=device,
        arguments=arguments,
        train_loader=train_loader,
        test_loader=test_loader,
        metrics=metrics,
        criterion=criterion,
        run_name=run_name
    )
    
    # training
    trainer.train()

    out(f"finishing at {get_date_stamp()}")


def log_start_run(arguments,out):
    arguments.PyTorch_version = torch.__version__
    arguments.PyThon_version = sys.version
    arguments.pwd = os.getcwd()
    out("PyTorch version:", torch.__version__, "Python version:", sys.version)
    out("Working directory: ", os.getcwd())
    out("CUDA avalability:", torch.cuda.is_available(), "CUDA version:", torch.version.cuda)
    out(arguments)   


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

@ex.automain
def run(arguments):
    metrics = Metrics()
    out = metrics.log_line
    print = out

    ensure_current_directory()
    log_start_run(arguments,out)
    out("\n\n")
    metrics._batch_size = arguments['batch_size']
    metrics._eval_freq = arguments['eval_freq']
    main(arguments,metrics)


