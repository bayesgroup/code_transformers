import argparse
import time
import os
os.environ['WANDB_MODE'] = 'dryrun'
from model import LightningModel
from src.utils.exp_utils import create_exp_dir

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

def main(args):
    seed_everything(args.seed)
    print(f"Experiment name: {args.name}")

    args.work_dir = os.path.join(args.work_dir, args.project, args.run_group, args.name,
        f"{time.strftime('%Y%m%d-%H%M%S')}")

    create_exp_dir(args.work_dir,
        scripts_to_save=["train.py", "model.py"], debug=args.debug)

    wandb_logger = WandbLogger(
        name=args.name,
        save_dir=args.work_dir,
        offline=True,
        entity=args.entity,
        project=args.project,
        group=args.run_group,
    ) if not args.debug and args.log_wandb else None

    tens_logger = TensorBoardLogger(
        save_dir=args.work_dir,
        name=args.name,
    ) if not args.debug and args.log_tb else None

    checkpoint_callback = ModelCheckpoint(
        filepath=args.work_dir,
        monitor="val/mrr_values",
        mode='max'
    )
    loggers = []
    if tens_logger is not None:
        loggers.append(tens_logger)
    if wandb_logger is not None:
        loggers.append(wandb_logger)

    model = LightningModel(args)
    
    # if wandb_logger is not None:
    #     wandb_logger.watch(model)

    args.n_all_param = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    print('=' * 100)
    for k, v in model.args.__dict__.items():
        print('    - {} : {}'.format(k, v))
    print('=' * 100)
    print(f'#params = {args.n_all_param}')

    trainer = pl.Trainer(
        default_root_dir=args.work_dir,
        resume_from_checkpoint=args.restart_cpt,
        checkpoint_callback=checkpoint_callback,
        progress_bar_refresh_rate=50,
        gpus=args.gpus, 
        num_nodes=args.nodes,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        distributed_backend='dp' if args.nodes == 1 else 'ddp',
        auto_scale_batch_size=False,
        deterministic=True, 
        logger=loggers if not args.debug else None,
        fast_dev_run=args.debug,
        gradient_clip_val=args.clip,
        limit_test_batches=args.max_eval_steps,
        limit_val_batches=args.max_eval_steps,
        limit_train_batches=args.max_train_steps
    )

    if not args.eval:
        trainer.fit(model)

    results = trainer.test(model)
    if not args.debug:
        if args.log_wandb:
            wandb_logger.log_metrics(results)
        if args.log_tb:
            tens_logger.log_metrics(results)


def add_train_arguments(parser):

    parser.add_argument('--base_dir', type=str, default='../data/processed_data_py',
                        help='location of the data corpus')
    parser.add_argument('--work_dir', default='GPT-CODE', type=str,
                        help='experiment directory.') 
    parser.add_argument('--project', type=str, 
                        default="code-prediction-transformer",
                        help='project name')
    parser.add_argument('--load', type=str, default='',
                        help='path to load weight')
    parser.add_argument('--name', type=str, default='N/A',
                        help='name of the trial')
    parser.add_argument('--debug', action='store_true',
                        help='run in debug mode (do not create exp dir)')
                   
    parser.add_argument('--log_wandb', action="store_true",
                        help='use Weights & Biases logger') 
    parser.add_argument('--run_group', type=str, 
                        default="experiment",
                        help='run_group name')
    parser.add_argument('--entity', type=str, default=None,
                        help='wandb teams name') 

    parser.add_argument('--log_tb', action="store_true",
                        help='use Tensorboard logger')     

    parser.add_argument('--preprocess', action='store_true',
                        help='to force converting dataset to vocab')                    
    parser.add_argument('--only_values', action='store_true',
                        help="use only non empty leaf values (no types)")   
    parser.add_argument('--use_test', action='store_true',
                        help='user test instead of validation data after each epoch (to make plots)')                       

    # Optimizers
    parser.add_argument('--optim', default='Adam', type=str,
                        choices=['Adam', 'SGD', 'Adagrad', 'RMSprop', 'RAdam'],
                        help='optimizer to use.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate (0.00025|5 for adam|sgd)')
    parser.add_argument('--scheduler', default='cosine', type=str,
                        choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                        help='lr scheduler to use.')
    parser.add_argument('--warmup_step', type=int, default=2000,
                        help='the number of steps to warm up the learning rate to its lr value')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='decay factor when ReduceLROnPlateau is used')
    parser.add_argument('--lr_min', type=float, default=0.0,
                        help='minimum learning rate during annealing')
    parser.add_argument('--eta_min', type=float, default=0.0,
                        help='min learning rate for cosine scheduler')

    # Gradient updates
    parser.add_argument('--clip', type=float, default=0.2,
                        help='gradient clipping')
    parser.add_argument('--clip_nonemb', action='store_true',
                        help='only clip the gradient of non-embedding params')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='accumulate grad batches')

    # Sequence logistics
    parser.add_argument('--ctx_len', type=int, default=500,
                        help='number of tokens in AST context')

    # Training techniques
    parser.add_argument('--seed', type=int, default=1111,
                        help='random_seed')
    parser.add_argument('--eval', action='store_true',
                        help='evaluation mode')
    parser.add_argument('--restart_cpt', type=str, default=None,
                        help='restart checkpoint')
    parser.add_argument('--gpus', type=int, default=1,
                        help='number of GPUs')
    parser.add_argument('--nodes', type=int, default=1,
                        help='number of nodes')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='the number of workers for DataLoader')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='report interval')
    parser.add_argument('--eval_interval', type=int, default=10000,
                        help='evaluation interval')             
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--max_eval_steps', type=int, default=1.0,
                        help='max eval steps')
    parser.add_argument('--max_train_steps', type=int, default=1.0,
                        help='max train steps')
    parser.add_argument('--start_train_steps', type=int, default=0,
                        help='starting training step count (default to 0)')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch GPT-2 for code generation', add_help=False)
    add_train_arguments(parser)
    
    parser = LightningModel.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
