"""
Submit a hyperparameter optimization to slurm using submitit and nevergrad.
"""
import os
import argparse

# from sklearn.model_selection import ParameterGrid
import submitit
import nevergrad as ng

from training.trainer import do_train
from misc.utils import TrainingParams, set_seed


def get_hyperparam_sweep(default_params: TrainingParams):
    """Define hyperparameter values to sweep through."""
    parametrization = ng.p.Instrumentation(
        params=default_params,
        lr=ng.p.Log(init=5e-4, lower=1e-4, upper=1e-2),
        octree_depth=ng.p.TransitionChoice([6,7,8]),
        model_params=ng.p.Dict(
            channels_blocks_top_down=ng.p.Choice([  # [[channels], [num_blocks], num_top_down]
                [[256], [18], 0],
                [[128,256], [10,10], 0],
                [[128,256], [2,18], 0],
                [[128,256,128], [8,8,8], 1],
                [[128,256,128], [2,18,2], 1],
                [[64,128,256], [2,2,18], 0],
                [[128,256,128,64], [2,18,2,2], 2],
                [[64,128,256,128], [2,2,18,2], 1],
                [[128,256,256,128,64], [2,10,10,2,2], 2],
            ]),
            patch_size=ng.p.TransitionChoice([16,32,64]),
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
    parser.add_argument('--budget', type=int, default=50,
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
    # TODO: compare optimizers
    optimizer = ng.optimizers.NgIohTuned(
        parametrization, budget=args.budget, num_workers=args.num_workers
    )
    
    # Configure executor
    # executor = submitit.AutoExecutor(folder=log_folder, cluster='debug')
    executor = submitit.AutoExecutor(folder=log_folder)
    job_days = 7
    executor.update_parameters(name="octf_hyperparam_sweep", timeout_min=job_days*24*60,
                               nodes=1, gpus_per_node=1, cpus_per_task=2,
                               tasks_per_node=1, slurm_mem="280gb",  # 280GB to prevent 2 jobs entering the same node (and having shm overflow)
                               slurm_mail_user="ethan.griffiths@data61.csiro.au",
                               slurm_mail_type="FAIL")
    # executor.map_array(do_train, params_list)  # pure submitit method

    # Optimize function
    recommendation = optimizer.minimize(do_train, executor=executor, verbosity=2)
    # recommendation = optimizer.minimize(debug_function, executor=executor, verbosity=2)

    print(f"\nFinal recommendation: {recommendation}")
    print(f"\nkwargs: {recommendation.kwargs}")
    