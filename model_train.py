from share import *
import os

from omegaconf import OmegaConf
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.model import create_model, load_state_dict
import torch
import argparse
from torch.utils.data import DataLoader
from functools import partial
import numpy as np

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config


# Configs
resume_path = './models/control_sd15_ini.ckpt'
# batch_size = 32
logger_freq = 300
learning_rate = 1e-4
sd_locked = True
only_mid_control = False


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train_batch_size=None, val_batch_size=None, train=None, validation=None, test=None, predict=None,
                 num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else 16
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        batch_size = self.train_batch_size if self.train_batch_size else self.batch_size
        return DataLoader(self.datasets["train"], batch_size=batch_size,
                          num_workers=self.num_workers, shuffle=False,
                          worker_init_fn=init_fn, persistent_workers=True)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        batch_size = self.val_batch_size if self.val_batch_size else self.batch_size
        return DataLoader(self.datasets["validation"],
                          batch_size=batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle, persistent_workers=True)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle, persistent_workers=True)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, persistent_workers=True)


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Meta Controlnet")

    parser.add_argument("-n", "--name", type=str, default="test",
        nargs="?", help="postfix for logdir")

    # parser.add_argument('--auto_path', type=str, default="",
    #                     help="pretrained auto-encoder path")
    
    parser.add_argument('--data_config', type=str, default="models/dataset_seg.yaml",
                        help="pretrained dataset path")
    
    parser.add_argument('--meta_method', type=str, default=None,
                        choices=['maml'], help='if we and how we use meta training')

    # parser.add_argument('--freeze', type=int, default=0, help='how many layers been frozen, \
    #                     from 1 to 4, 1 means we only freeze middle block, 4 means only finetune first encoder')
    
    parser.add_argument('--resume_path', type=str, default=None, help='where we load checkpoint')

    parser.add_argument('--lr', type=float, default=None, help='learning rate for training')

    parser.add_argument('--eval', action='store_true', help='if we evaluate the model')

    parser.add_argument('--maml_freeze', type=str, default=None,
                            help='how we change the maml_freeze')

    parser.add_argument('--num_inner_steps', type=int, default=1, help='inner step number')

    parser.add_argument('--train_batch_size', type=int, default=None, help='training batch size')

    parser.add_argument('--inner_batch_size', type=int, default=None, help='maml inner step training batch size')

    parser.add_argument('--inner_freeze_only', action='store_true', help='if we only freeze when inner training')

    parser.add_argument('--eval_mode', type=str, default='finetune', choices=['fewshot', 'finetune', 'fewtrain'], 
                        help='how we evaluate the model')

    parser.add_argument('--val_only', action='store_true', help='if we only check the validation performance')

    parser.add_argument('--save_no_indiv', action='store_true', help='if we save pictures together')


    # Parse the arguments
    args = parser.parse_args()
    opt, _ = parser.parse_known_args()

    nowname = f"{opt.name}"
    logdir = os.path.join('logdir', nowname)
    ckptdir = os.path.join(logdir, "checkpoints")

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)

    resume_path = './models/control_sd15_ini.ckpt'
    if args.meta_method == 'maml':
        model = create_model('./models/maml_cldm_v15.yaml').cpu()
    else:
        model = create_model('./models/cldm_v15.yaml').cpu()
    if args.resume_path:
        print(args.resume_path)
        resume_path = args.resume_path
    model.load_state_dict(load_state_dict(resume_path, location='cuda:0'))

    control_state_dict = model.state_dict()

    # if len(args.auto_path) == 0:
    #     print('vanilla control net')
    # else:
    #     # load pretrained autoencoder checkpoint
    #     auto_checkpoint = torch.load(args.auto_path)
    #     auto_state_dict = auto_checkpoint['state_dict']

    #     for name, param in control_state_dict.items():
    #         if name.startswith("first_stage_model."):
    #             auto_name = name[len("first_stage_model."):]
    #             if auto_name in auto_state_dict:
    #                 control_state_dict[name] = auto_state_dict[auto_name]
    #             else:
    #                 print(f"Warning: {auto_name} not found")
    #                 raise NotImplementedError
    #     model.load_state_dict(control_state_dict)
    
    # check meta method
    if args.maml_freeze is None:
        # default block list
        block_list = ['asdfg']
    elif args.maml_freeze == 'block_9_12':
        block_list = [f"control_model.input_blocks.{i}" for i in range(9, 12)]
        block_list.append("control_model.middle_block")
    else:
        block_list = ['asdfg']

    for name, param in model.named_parameters():
        if any(block_name in name for block_name in block_list):
            param.requires_grad = False
        else:
            param.requires_grad = True
        

    lr = args.lr if args.lr else learning_rate
    model.learning_rate = lr
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    model.maml_freeze = args.maml_freeze
    model.num_inner_steps = args.num_inner_steps
    model.inner_batch_size = args.inner_batch_size

    # Data
    data_config = OmegaConf.load(opt.data_config)
    dataloader = instantiate_from_config(data_config.data)
    if args.train_batch_size:
        dataloader.train_batch_size = args.train_batch_size
    dataloader.prepare_data()
    dataloader.setup()
    print("#### Data #####")
    for k in dataloader.datasets:
        print(f"{k}, {dataloader.datasets[k].__class__.__name__}, {len(dataloader.datasets[k])}")

    batch_frequency = 500
    save_indiv = False 
    if args.eval:
        batch_frequency = 1
        save_indiv = True
        if args.save_no_indiv:
            save_indiv = False
    # Callbacks
    callbacks_cfg = {
        "checkpoint_callback": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}-{step:09}",
                "verbose": True,
                'save_top_k': -1,
                'every_n_train_steps': 1000,
                'save_weights_only': True,
                "save_last": True,
            }
        },
        "image_logger": {
            "target": "cldm.logger.ImageLogger",
            "params": {
                "batch_frequency": batch_frequency,
                "max_images": 32,
                "clamp": True,
                "log_images_kwargs": {'N': 32,
                                      'unconditional_guidance_scale': 9.0},
                "save_indiv": save_indiv,
            }
        },
    }

    callbacks = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    gpus_count = torch.cuda.device_count()

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=logdir)
    if not args.eval:
        trainer = pl.Trainer(gpus=gpus_count, accelerator='ddp',
                        max_steps=100000, check_val_every_n_epoch=1, accumulate_grad_batches=4,
                        precision=32, callbacks=callbacks, logger=tb_logger)

    elif args.eval:
        if args.eval_mode == 'finetune':
            trainer = pl.Trainer(gpus=gpus_count, accelerator='ddp',
                        check_val_every_n_epoch=2, accumulate_grad_batches=8,
                        precision=32, callbacks=callbacks, logger=tb_logger,
                        limit_train_batches=8, limit_val_batches=1, max_epochs=250)
        elif args.eval_mode == 'fewshot':
            trainer = pl.Trainer(gpus=1, accelerator='ddp',
                        check_val_every_n_epoch=1, accumulate_grad_batches=1,
                        precision=32, callbacks=callbacks, logger=tb_logger,
                        limit_train_batches=1, limit_val_batches=1, max_epochs=30)
            
        elif args.eval_mode == 'fewtrain':
            trainer = pl.Trainer(gpus=gpus_count, accelerator='ddp',
                        check_val_every_n_epoch=100, accumulate_grad_batches=4,
                        precision=32, callbacks=callbacks, logger=tb_logger,
                        limit_train_batches=8, limit_val_batches=1, max_epochs=2000)


    # check performance first
    if args.val_only:
        trainer.validate(model, dataloader)
    # Train!
    else:
        trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
