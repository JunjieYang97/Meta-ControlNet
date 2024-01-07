# Meta ControlNet Offical Code
Codes for paper [Meta ControlNet: Enhancing Task Adaptation via Meta Learning](https://arxiv.org/abs/2312.01255) by Junjie Yang, Jinze Zhao, Peihao Wang, Zhangyang Wang, Yingbin Liang.

## How to use our code.
This repo is built on [ControlNet](https://github.com/lllyasviel/ControlNet) and  we demonstrate our proposed algorithm enjoys faster convergence and better generalization abilites compared with ControlNet. To use our code, please refer to above ControlNet repo for its environment dependecies. We have tested this code in Python 3.8 with PyTorch 2.0.1, CUDA 10.2.

### Detailed instructions.

We introduce the following hyperparameters to run the code in [model_train.py](https://github.com/JunjieYang97/Meta-ControlNet/blob/master/model_train.py).


+ `--data_config`: Specify the task we should use. For example, we use the "models/dataset_maml_train.yaml" for training, and "models/dataset_seg.yaml" for testing. You can modify the task_list in yaml file to specify the task you need to train or evaluate.
+ `--meta_method`: Have to specify it as "maml" if we wanna meta training, otherwise, train the model with vanilla ControlNet.
+ `--resume_path`: Loading the saved model checkpoint. For example, "--resume_path logdir/maml_train/checkpoints/epoch=000028-step=000007999.ckpt"
+ `--eval`: If it is true, then we begin to finetune the trained checkpoint in new tasks. 
+ `--maml_freeze`: Specify how we freeze the layers in U-net. During the training, we should set it as "block_9_12".
+ `--eval_mode`: How we further finetune the model. "fewshot" means we only update with few samples. "finetune" means we only update with full batch size but few steps. Usually, "fewtrain" is not required which refers to longer time training.
+ `--val_only`: Only evaluation without finetuning.


We provide following commands to run the code. Generally, we should meta train the model with following command:

#### Meta train with three tasks:

```python
python model_train.py --name maml_train_9_12 --meta_method maml --data_config models/dataset_maml_train.yaml \
    --maml_freeze block_9_12
```

Then you can few-shot finetune in the edge-based tasks. Note that, you can also choose to direct evaluate the checkpoint without finetuning.

#### Few-shot finetuning in new task Normal:
```python
python model_train.py --eval --name normal_9_12 --data_config models/dataset_normal.yaml --resume_path logdir/maml_train_9_12/checkpoints/epoch=000028-step=000007999.ckpt --eval_mode fewshot
```

#### Directly evaluate in new task Normal:
```python
python model_train.py --name normal_9_12_eval --data_config models/dataset_normal.yaml --resume_path logdir/maml_train_9_12/checkpoints/epoch=000028-step=000007999.ckpt --val_only
```

For non-edge task, e.g., openpose, we use the following finetune command:
#### Finetuning in new task Openpose:
```python
python model_train.py --eval --name openpose_nofreeze --data_config models/dataset_openpose.yaml --resume_path logdir/maml_train_9_12/checkpoints/epoch=000028-step=000007999.ckpt --train_batch_size 64 --eval_mode finetune
```


## Citation

If this repo is useful for your research, please cite our paper:

```tex
@article{yang2023meta,
  title={Meta ControlNet: Enhancing Task Adaptation via Meta Learning},
  author={Yang, Junjie and Zhao, Jinze and Wang, Peihao and Wang, Zhangyang and Liang, Yingbin},
  journal={arXiv preprint arXiv:2312.01255},
  year={2023}
}
```

