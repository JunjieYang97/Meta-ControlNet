import json
import cv2
import numpy as np
import math
from pathlib import Path
import torch
import os
from PIL import Image
from typing import List

from torch.utils.data import Dataset


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

class MyDataset(Dataset):
    def __init__(self,
        path: str,
        split: str = "train",
        splits: [float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        task_list: List[str] = ["seg"],
        batch_size: int = None,
        meta_method: str = None,
        prompt_option: str = None
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.split = split
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.task_list = task_list
        self.batch_size = batch_size
        self.meta_method = meta_method
        self.prompt_option = prompt_option

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        if task_list[0] == "openpose" or task_list[0] == "inv_openpose":
            with open(Path(self.path, "openpose_both_seeds.json")) as f:
                self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]
        # self.data = []
        # with open('./training/fill50k/prompt.json', 'rt') as f:
        #     for line in f:
        #         self.data.append(json.loads(line))


    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, idx):
        # item = self.data[idx]

        # source_filename = item['source']
        # target_filename = item['target']
        # prompt = item['prompt']

        # source = cv2.imread('./training/fill50k/' + source_filename)
        # target = cv2.imread('./training/fill50k/' + target_filename)

        # # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # # Normalize source images to [0, 1].
        # source = source.astype(np.float32) / 255.0

        # # Normalize target images to [-1, 1].
        # target = (target.astype(np.float32) / 127.5) - 1.0

        # return dict(jpg=target, txt=prompt, hint=source)


        name, seeds = self.seeds[idx]
        propt_dir = Path(self.path, name)
        seed = seeds[torch.randint(0, len(seeds), ()).item()]
        if self.split == 'val':
            seed = seeds[0]

        prompt_text = 'input'
        imageseq = '0'
        if self.prompt_option == 'output':
            prompt_text = 'output'
            imageseq = '1'

        # Load text prompt
        with open(propt_dir.joinpath("prompt.json")) as fp:
            prompt = json.load(fp)
            prompt = prompt[prompt_text]

        # Load input and output images; shape -> h w c
        task = np.random.choice(self.task_list)
        if self.split == 'val':
            task = self.task_list[idx%len(self.task_list)]


        if task.startswith("inv_"):
            task = task[4:]
            image_path_target = f"{seed}_{imageseq}_{task}.jpg"
            image_path_source = f"{seed}_{imageseq}.jpg"
            prompt = task+' map'          
        else:
            image_path_target = f"{seed}_{imageseq}.jpg"
            image_path_source = f"{seed}_{imageseq}_{task}.jpg"

        # while not os.path.exists(propt_dir.joinpath(image_path_target)):
        #     i = i+1
        #     name, seeds = self.seeds[i]
        #     propt_dir = Path(self.path, name)
        #     seed = seeds[torch.randint(0, len(seeds), ()).item()]
            
        #     image_path_target = f"{seed}_0.jpg"
        #     image_path_source = f"{seed}_0_{task}.jpg"


        image_target = Image.open(propt_dir.joinpath(image_path_target))
        image_source = Image.open(propt_dir.joinpath(image_path_source))

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        
        image_target = image_target.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_source = image_source.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_source = 2 * torch.tensor(HWC3(np.array(image_source))).float() / 255. -1
        image_target = 2 * torch.tensor(HWC3(np.array(image_target))).float() / 255. -1
        # image_target = torch.tensor(HWC3(np.array(image_target))).float() / 127.5 - 1


        if self.meta_method == 'maml':
            return dict(jpg=image_target, txt=prompt, hint=image_source, task=task)
        else:
            return dict(jpg=image_target, txt=prompt, hint=image_source)

        # image_0 = 2 * torch.tensor(np.array(image_0)).float() / 255. - 1
        # image_1 = 2 * torch.tensor(np.array(image_1)).float() / 255. - 1

        # if image_0.dim() == 2:
        #     image_0 = image_0.unsqueeze(-1)

        # print(task)
        # print(image_0.shape)

        # return dict(image=image_0)

        # Load Controls; shape -> h w c

        # task = np.random.choice(['inv_seg', 'inv_depth', 'inv_hed', 'seg', 'depth', 'hed'])
        # txt_log = task
        # if task == 'inv_seg':
        #     image_seg = Image.open(propt_dir.joinpath(f"{example_seed}_0_seg.jpg"))
        #     image_seg = image_seg.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        #     image_seg = 2 * torch.tensor(np.array(image_seg)).float() / 255. - 1

        #     example_pair = torch.cat((image_seg, image_0), dim=2) # h w c

        #     image_query = Image.open(propt_dir.joinpath(f"{seed}_1_seg.jpg"))
        #     image_query = image_query.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        #     image_query = 2 * torch.tensor(np.array(image_query)).float() / 255. - 1

        #     image_target = image_1

        # elif task == 'seg':
        #     image_seg = Image.open(propt_dir.joinpath(f"{example_seed}_0_seg.jpg"))
        #     image_seg = image_seg.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        #     image_seg = 2 *  torch.tensor(np.array(image_seg)).float() / 255. - 1

        #     example_pair = torch.cat((image_0, image_seg), dim=2)  # h w c
        #     image_query = image_1
        #     prompt = 'segmentation map'

        #     image_target = Image.open(propt_dir.joinpath(f"{seed}_1_seg.jpg"))
        #     image_target = image_target.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        #     image_target = 2 * torch.tensor(np.array(image_target)).float() / 255. - 1

        # elif task == 'inv_depth':
        #     image_depth = Image.open(propt_dir.joinpath(f"{example_seed}_0_depth.jpg"))
        #     image_depth = image_depth.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        #     image_depth = 2 * torch.tensor(HWC3(np.array(image_depth))).float() / 255. - 1

        #     example_pair = torch.cat((image_depth, image_0), dim=2)  # h w c

        #     image_query = Image.open(propt_dir.joinpath(f"{seed}_1_depth.jpg"))
        #     image_query = image_query.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        #     image_query = 2 * torch.tensor(HWC3(np.array(image_query))).float() / 255. - 1

        #     image_target = image_1

        # elif task == 'depth':
        #     image_depth = Image.open(propt_dir.joinpath(f"{example_seed}_0_depth.jpg"))
        #     image_depth = image_depth.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        #     image_depth = 2 * torch.tensor(HWC3(np.array(image_depth))).float() / 255. - 1

        #     example_pair = torch.cat((image_0, image_depth), dim=2)  # h w c
        #     image_query = image_1
        #     prompt = 'depth map'

        #     image_target = Image.open(propt_dir.joinpath(f"{seed}_1_depth.jpg"))
        #     image_target = image_target.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        #     image_target = 2 * torch.tensor(HWC3(np.array(image_target))).float() / 255. - 1

        # elif task == 'inv_hed':

        #     image_hed = Image.open(propt_dir.joinpath(f"{example_seed}_0_hed.jpg"))
        #     image_hed = image_hed.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        #     image_hed = 2 * torch.tensor(HWC3(np.array(image_hed))).float() / 255. - 1

        #     example_pair = torch.cat((image_hed, image_0), dim=2)  # h w c

        #     image_query = Image.open(propt_dir.joinpath(f"{seed}_1_hed.jpg"))
        #     image_query = image_query.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        #     image_query = 2 * torch.tensor(HWC3(np.array(image_query))).float() / 255. - 1

        #     image_target = image_1

        # elif task == 'hed':

        #     image_hed = Image.open(propt_dir.joinpath(f"{example_seed}_0_hed.jpg"))
        #     image_hed = image_hed.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        #     image_hed = 2 * torch.tensor(HWC3(np.array(image_hed))).float() / 255. - 1

        #     example_pair = torch.cat((image_0, image_hed), dim=2)  # h w c
        #     image_query = image_1
        #     prompt = 'hed map'

        #     image_target = Image.open(propt_dir.joinpath(f"{seed}_1_hed.jpg"))
        #     image_target = image_target.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        #     image_target = 2 * torch.tensor(HWC3(np.array(image_target))).float() / 255. - 1


        # return dict(jpg=image_target, txt=prompt, query=image_query, example_pair=example_pair, txt_log=txt_log)


