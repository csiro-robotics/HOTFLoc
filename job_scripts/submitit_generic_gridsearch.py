"""
Submit a generic single node training job using submitit.
"""
import os.path as osp
import argparse

import yaml
import submitit
from sklearn.model_selection import ParameterGrid

from training.trainer import do_train
from misc.utils import TrainingParams, set_seed

def get_hyperparam_grid(base_params: TrainingParams, gridsearch_config_file: str):
    # Load gridsearch config and generate parameter grid
    assert osp.isfile(gridsearch_config_file), "Invalid gridsearch config provided"
    with open(gridsearch_config_file, 'r') as file:
        gridsearch_dict = yaml.safe_load(file)
    if 'model_params' in gridsearch_dict:  # convert nested dict to param grid
        model_params_dict = gridsearch_dict.pop('model_params')
        gridsearch_dict['model_params'] = list(ParameterGrid(model_params_dict))
    hyperparam_grid = list(ParameterGrid(gridsearch_dict))
    # Append gridsearch params to base config for every job submitted
    # hyperparam_grid = [(base_params, grid_params) for grid_params in hyperparam_grid]
    return hyperparam_grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Submit a training job through submitit.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to base configuration file')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the base model configuration file')
    parser.add_argument('--gridsearch_config', type=str, required=True,
                        help='Path to the gridsearch configuration file')
    parser.add_argument('--log_folder', type=str, default='submitit_logs',
                        help='Path to store submitit logs and pickles')
    parser.add_argument('--mem', type=str, default='220gb',
                        help='Memory requested per job')
    parser.add_argument('--num_workers', type=int, default=30,
                        help='Num jobs that can run in parallel at once')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode, submit jobs on local device')
    args = parser.parse_args()
    print('Base config path: {}'.format(args.config))
    print('Base model config path: {}'.format(args.model_config))
    print('Log folder: {}'.format(args.log_folder))

    job_days = 7
    job_config = {
        'nodes': 1, 'gpus_per_node': 1, 'cpus_per_task': 3, 'slurm_mem': args.mem,  # 280GB to prevent 2 jobs entering the same node (and having shm overflow)
        'timeout_min': job_days*24*60, 'slurm_mail_user': 'ethan.griffiths@data61.csiro.au',
        'slurm_mail_type': 'FAIL', 'slurm_gres': 'one:1', 
        'slurm_array_parallelism': args.num_workers,
        # 'slurm_dependency': 'afterany:57846199'
    }
    
    # Seed RNG
    set_seed()
    # Add job name to log files
    log_folder = osp.join(args.log_folder, '%j')

    base_params = TrainingParams(args.config, args.model_config, debug=args.debug)

    hyperparam_grid = get_hyperparam_grid(base_params, args.gridsearch_config)
    
    # Configure executor
    cluster = 'debug' if args.debug else None
    executor = submitit.AutoExecutor(folder=log_folder, cluster=cluster)
    executor.update_parameters(name=base_params.model_params.model, **job_config)
    jobs = executor.map_array(do_train, [base_params]*len(hyperparam_grid), hyperparam_grid)

    print(f"Jobs submitted")

    # NOTE: Can wait for jobs to return using the below commented out line
    # output = jobs[0].result()

    