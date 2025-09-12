# Warsaw University of Technology
# Train MinkLoc model

import argparse
import torch

from training.trainer import NetworkTrainer
from misc.utils import TrainingParams


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MinkLoc3Dv2 model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the model-specific configuration file')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from the given checkpoint. Ensure config and model_config matches the supplied checkpoint.')
    parser.add_argument('--finetune_from', type=str, default=None,
                        help='Finetune from the given checkpoint. Ensure config and model_config matches the supplied checkpoint.')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging')

    args = parser.parse_args()
    print('Training config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    assert not (args.resume_from is not None and args.finetune_from is not None), (
        'Conflicting options, cannot resume training and finetune'
    )
    if args.resume_from is not None:
        print('Resuming from checkpoint path: {}'.format(args.resume_from))
    if args.finetune_from is not None:
        print('Finetuning from checkpoint path: {}'.format(args.finetune_from))
    print('Debug mode: {}'.format(args.debug))
    print('Verbose mode: {}'.format(args.verbose))
    print('Disable wandb: {}'.format(args.disable_wandb))
    
    params = TrainingParams(args.config, args.model_config,
                            debug=args.debug, verbose=args.verbose)
    params.wandb = not args.disable_wandb

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    training_callable = NetworkTrainer()
    training_callable(
        params,
        checkpoint_path=args.resume_from,
        finetune_path=args.finetune_from,
    )
