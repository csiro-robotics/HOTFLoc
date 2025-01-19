"""
Submit a generic single node training job using submitit.
"""
import os
import argparse
import time

import submitit

from training.trainer import NetworkTrainer
from misc.utils import TrainingParams, set_seed

job_config = {
    'nodes': 1, 'gpus_per_node': 1, 'cpus_per_task': 4,
    'slurm_mail_user': 'ethan.griffiths@data61.csiro.au',
    'slurm_mail_type': 'FAIL,END',
    # 'slurm_exclude': 'g088',
    # 'slurm_gres': 'one:1',
    # 'slurm_dependency': 'afterany:1438093',
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Submit a training job through submitit.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to base configuration file')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the base model configuration file')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from the given checkpoint. Ensure config and model_config matches the supplied checkpoint.')
    parser.add_argument('--log_folder', type=str, default='submitit_logs',
                        help='Path to store submitit logs and pickles')
    parser.add_argument('--job_days', type=float, default=7.0,
                        help='Number of days to request a job for. Accepts decimal values, min 2 hours')
    parser.add_argument('--job_mem', type=str, default='200gb',
                        help='Memory requested per job, 280gb to prevent 2 jobs entering the same node (and causing shm overflow)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode, submit jobs on local device')
    args = parser.parse_args()
    print('Base config path: {}'.format(args.config))
    print('Base model config path: {}'.format(args.model_config))
    if args.resume_from is not None:
        print('Resuming from checkpoint path: {}'.format(args.resume_from))
    print('Log folder: {}'.format(args.log_folder))
    print('Days requested: {}'.format(args.job_days))
    print('Mem requested: {}'.format(args.job_mem))
    print('Debug: {}'.format(args.debug))

    # Update job request
    job_config['timeout_min'] = round(args.job_days*24*60)
    job_config['slurm_mem'] = args.job_mem
    
    # Seed RNG
    set_seed()
    # Add job name to log files
    log_folder = os.path.join(args.log_folder, '%j')

    params = TrainingParams(args.config, args.model_config, debug=False)
    
    # Configure executor
    cluster = 'debug' if args.debug else None
    executor = submitit.AutoExecutor(
        folder=log_folder, cluster=cluster,
        slurm_max_num_timeout=5,
    )
    executor.update_parameters(name=params.model_params.model, **job_config)
    training_callable = NetworkTrainer()
    job = executor.submit(training_callable, params, checkpoint_path=args.resume_from)
    # executor.map_array(do_train, params_list)  # pure submitit method

    print(f"Job {job.job_id} submitted")

    # NOTE: Can wait for job to return using the below commented out line
    # output = job.result()

    # # Interrupt job early to test ckpt handling
    # sleep_mins = 5
    # time.sleep(sleep_mins*60)
    # job._interrupt(timeout=False)  # preemption
    # # job._interrupt(timeout=True)  # timeout
    # print('INTERRUPTED')

    