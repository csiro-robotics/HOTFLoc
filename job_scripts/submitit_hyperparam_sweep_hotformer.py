"""
Submit a hyperparameter optimization to slurm using submitit and nevergrad.
"""
import os
import argparse
import time

# from sklearn.model_selection import ParameterGrid
import submitit
import nevergrad as ng
import torch

from training.trainer import do_train
from misc.utils import TrainingParams, set_seed

def debug_function(params: TrainingParams = None, *args, **kwargs):
    # Update params with hyperparam sweep
    for key, value in kwargs.items():
        if key != 'model_params':
            setattr(params, key, value)
            continue
        for model_key, model_value in value.items():
            setattr(params.model_params, model_key, model_value)

    time.sleep(5)

    assert torch.cuda.is_available(), "CUDA not available!"    
    return 1+1

def get_hyperparam_sweep(default_params: TrainingParams):
    """Define hyperparameter values to sweep through."""
    parametrization = ng.p.Instrumentation(
        params=default_params,
        lr=ng.p.Log(init=1e-4, lower=1e-4, upper=1e-2),
        octree_depth=ng.p.TransitionChoice([7, 8, 9]),
        weight_decay=ng.p.Log(init=1e-4, lower=1e-4, upper=1e-1),
        tau1=ng.p.Log(init=1e-2, lower=1e-3, upper=1e-1),
        model_params=ng.p.Dict(
            channels=ng.p.TransitionChoice(
                [[128,256,128,64], [64,128,256,128], [32,64,128,256]]),
            num_blocks=ng.p.TransitionChoice([2,6,18], repetitions=4),
            ct_layers=ng.p.Tuple(  # 2nd layer always has CTs enabled
                ng.p.Choice([False, True]), True,
                ng.p.Choice([False, True]), ng.p.Choice([False, True])),
            ct_propagation_scale=ng.p.Log(init=1e-1, lower=1e-2, upper=5e-1),
            use_ADaPE=ng.p.Choice([False, True]),
            num_top_down=ng.p.TransitionChoice([0, 1, 2]),
            patch_size=ng.p.TransitionChoice([16,32,64]),
            conv_norm=ng.p.Choice(['batchnorm', 'layernorm'])
        )
    )
    return parametrization

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Setup a hyperparameter search using SLURM.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to base configuration file')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the base model configuration file')
    parser.add_argument('--log_folder', type=str, default='submitit_logs',
                        help='Path to store submitit logs and pickles')
    parser.add_argument('--budget', type=int, default=120,
                        help='Max number of jobs to submit for hyperparam tuning.')
    parser.add_argument('--num_workers', type=int, default=20,
                        help='Max number of jobs to have running at once.')
    args = parser.parse_args()
    print('Base config path: {}'.format(args.config))
    print('Base model config path: {}'.format(args.model_config))
    print('Log folder: {}'.format(args.log_folder))
    
    # Seed RNG
    set_seed()
    # Add job name to log files
    log_folder = os.path.join(args.log_folder, '%j')

    default_params = TrainingParams(args.config, args.model_config, debug=False)
    parametrization = get_hyperparam_sweep(default_params)
    # TODO: compare optimizers, but this one seems suitable for DL models with
    # a mix of discrete and continuous parameters
    optimizer = ng.optimizers.PortfolioDiscreteOnePlusOne(
        parametrization, budget=args.budget, num_workers=args.num_workers
    )
    
    # Configure executor
    # executor = submitit.AutoExecutor(folder=log_folder, cluster='debug')
    executor = submitit.AutoExecutor(folder=log_folder)
    job_days = 7
    executor.update_parameters(name="hotf_hyperparam_sweep", timeout_min=job_days*24*60,
                               nodes=1, gpus_per_node=1, cpus_per_task=3,
                               tasks_per_node=1, slurm_mem="180gb",
                               slurm_mail_user="ethan.griffiths@data61.csiro.au",
                               slurm_mail_type="FAIL")
    # executor.map_array(do_train, params_list)  # pure submitit method

    # Optimize function
    recommendation = optimizer.minimize(do_train, executor=executor, verbosity=2)
    # recommendation = optimizer.minimize(debug_function, executor=executor, verbosity=2)

    print(f"\nFinal recommendation: {recommendation}")
    print(f"\nkwargs: {recommendation.kwargs}")
    