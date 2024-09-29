"""
Submit a generic single node training job using submitit.
"""
import os
import argparse

import submitit

from training.trainer import do_train
from misc.utils import TrainingParams, set_seed

job_days = 7
job_config = {
    'nodes': 1, 'gpus_per_node': 1, 'cpus_per_task': 3, 'slurm_mem': '220gb',  # 280GB to prevent 2 jobs entering the same node (and having shm overflow)
    'timeout_min': job_days*24*60, 'slurm_mail_user': 'ethan.griffiths@data61.csiro.au',
    'slurm_mail_type': 'FAIL,END', 'slurm_gres': 'one:1', 
    # 'slurm_dependency': 'afterany:57846199'
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Submit a training job through submitit.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to base configuration file')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the base model configuration file')
    parser.add_argument('--log_folder', type=str, default='submitit_logs',
                        help='Path to store submitit logs and pickles')
    args = parser.parse_args()
    print('Base config path: {}'.format(args.config))
    print('Base model config path: {}'.format(args.model_config))
    print('Log folder: {}'.format(args.log_folder))
    
    # Seed RNG
    set_seed()
    # Add job name to log files
    log_folder = os.path.join(args.log_folder, '%j')

    params = TrainingParams(args.config, args.model_config, debug=False)
    
    # Configure executor
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(name=params.model_params.model, **job_config)
    job = executor.submit(do_train, params)
    # executor.map_array(do_train, params_list)  # pure submitit method

    print(f"Job {job.job_id} submitted")

    # NOTE: Can wait for job to return using the below commented out line
    # output = job.result()

    