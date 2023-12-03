import einops
import torch
import torch as th
import torch.nn as nn

from cldm.cldm import ControlLDM
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from copy import deepcopy


class MAML_ControlLDM(ControlLDM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_inner_steps = 1
        self.task_num = 3

    def construct_meta_data(self, batch):
        seg_data_inner, depth_data_inner, hed_data_inner = {'txt': []}, {'txt': []}, {'txt': []}
        seg_data_val, depth_data_val, hed_data_val = {'txt': []}, {'txt': []}, {'txt': []}

        seg_jpgs, seg_hints, depth_jpgs, depth_hints, hed_jpgs, hed_hints = [], [], [], [], [], []
        seg_jpgs_val, seg_hints_val, depth_jpgs_val, depth_hints_val, hed_jpgs_val, hed_hints_val = [], [], [], [], [], []

        seg_count, depth_count, hed_count = 0, 0, 0

        for i, task in enumerate(batch['task']):
            if self.inner_batch_size:
                # For inner data
                if task == 'seg' and seg_count < self.inner_batch_size:
                    seg_data_inner['txt'].append(batch['txt'][i])
                    seg_jpgs.append(batch['jpg'][i])
                    seg_hints.append(batch['hint'][i])
                    seg_count += 1
                elif task == 'depth' and depth_count < self.inner_batch_size:
                    depth_data_inner['txt'].append(batch['txt'][i])
                    depth_jpgs.append(batch['jpg'][i])
                    depth_hints.append(batch['hint'][i])
                    depth_count += 1
                elif task == 'hed' and hed_count < self.inner_batch_size:
                    hed_data_inner['txt'].append(batch['txt'][i])
                    hed_jpgs.append(batch['jpg'][i])
                    hed_hints.append(batch['hint'][i])
                    hed_count += 1
            
            # For validation data
            if task == 'seg':
                seg_data_val['txt'].append(batch['txt'][i])
                seg_jpgs_val.append(batch['jpg'][i])
                seg_hints_val.append(batch['hint'][i])
            elif task == 'depth':
                depth_data_val['txt'].append(batch['txt'][i])
                depth_jpgs_val.append(batch['jpg'][i])
                depth_hints_val.append(batch['hint'][i])
            elif task == 'hed':
                hed_data_val['txt'].append(batch['txt'][i])
                hed_jpgs_val.append(batch['jpg'][i])
                hed_hints_val.append(batch['hint'][i])

        if self.inner_batch_size:
            # For inner data
            if seg_jpgs:
                seg_data_inner['jpg'] = torch.stack(seg_jpgs)
                seg_data_inner['hint'] = torch.stack(seg_hints)
            if depth_jpgs:
                depth_data_inner['jpg'] = torch.stack(depth_jpgs)
                depth_data_inner['hint'] = torch.stack(depth_hints)
            if hed_jpgs:
                hed_data_inner['jpg'] = torch.stack(hed_jpgs)
                hed_data_inner['hint'] = torch.stack(hed_hints)
            task_batch_inner = [seg_data_inner, depth_data_inner, hed_data_inner]

        # For vanilla data
        if seg_jpgs_val:
            seg_data_val['jpg'] = torch.stack(seg_jpgs_val)
            seg_data_val['hint'] = torch.stack(seg_hints_val)
        if depth_jpgs_val:
            depth_data_val['jpg'] = torch.stack(depth_jpgs_val)
            depth_data_val['hint'] = torch.stack(depth_hints_val)
        if hed_jpgs_val:
            hed_data_val['jpg'] = torch.stack(hed_jpgs_val)
            hed_data_val['hint'] = torch.stack(hed_hints_val)

        task_batch_val = [seg_data_val, depth_data_val, hed_data_val]

        if self.inner_batch_size:
            return task_batch_inner, task_batch_val

        return task_batch_val, task_batch_val


    def shared_step(self, batch, params=None, return_grads=False, **kwargs):
        # replace the model params with given params if possible
        if params is not None:
            model_params = [param for param in self.control_model.parameters() if param.requires_grad]
            for param, new_param in zip(model_params, params):
                param.data = new_param.data
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        if return_grads:
            val_loss, val_loss_dict = loss
            meta_grads = torch.autograd.grad(val_loss, model_params, retain_graph=True, allow_unused=True)
            return val_loss, val_loss_dict, meta_grads
        return loss

    def inner_update(self, batch, params=None):
        if params is None:
            params = [param for param in self.control_model.parameters() if param.requires_grad]
        
        # Compute loss with respect to task
        loss, loss_dict, grads = self.shared_step(batch, params=params, return_grads=True)
        
        # Update model parameters
        if self.maml_freeze is not None:
            updated_params = [param if (grad is None or any(block_name in name for block_name in self.block_list)) else param - self.lr_inner * grad 
                              for param, grad, name in zip(params, grads, self.name_list)]
        else:
            updated_params = [param if grad is None else param - self.lr_inner * grad for param, grad in zip(params, grads)]
        return updated_params
    
    def training_step(self, batch, batch_idx):
        task_batch_inner, task_batch_val = self.construct_meta_data(batch)

        original_parameters = [param.clone() for param in self.control_model.parameters() if param.requires_grad]

        self.lr_inner = self.learning_rate
        if self.maml_freeze is not None:
            self.name_list = [name for name, param in self.control_model.named_parameters() if param.requires_grad]
            if self.maml_freeze == 'block_9_12':
                self.block_list = [f"control_model.input_blocks.{i}" for i in range(9, 12)]
                self.block_list.append("control_model.middle_block")

        meta_loss = 0
        meta_grads_list = []
        for i, (batch_inner, batch_val) in enumerate(zip(task_batch_inner, task_batch_val)):
            updated_params = original_parameters
            for _ in range(self.num_inner_steps):
                updated_params = self.inner_update(batch_inner, updated_params)

            # Compute validation loss on updated model
            val_loss, val_loss_dict, meta_grads = self.shared_step(batch_val, params=updated_params, return_grads=True)
            meta_loss += val_loss

            meta_grads_list.append(meta_grads)
            
        meta_loss = meta_loss / len(task_batch_val)
        averaged_meta_grads = []
        for grads in zip(*meta_grads_list):
            if all(g is None for g in grads):  # If all gradients are None
                averaged_meta_grads.append(None)
            else:
                averaged_meta_grads.append(sum(grads)/len(grads))

        index_tmp = 0
        for index, param in enumerate(list(self.control_model.parameters())):
            if param.requires_grad:
                updated_grads = averaged_meta_grads[index_tmp]
                index_tmp += 1
                param.grad = updated_grads

        self.log_dict(val_loss_dict, prog_bar=True,
                        logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                    prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        return meta_loss


    def configure_optimizers(self):
        lr = self.learning_rate
        params = [param for param in self.control_model.parameters() if param.requires_grad]
        
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
