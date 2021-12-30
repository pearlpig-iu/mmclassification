<div align="center">
  <img src="resources/mmcls-logo.png" width="600"/>
</div>

[![Build Status](https://github.com/open-mmlab/mmclassification/workflows/build/badge.svg)](https://github.com/open-mmlab/mmclassification/actions)
[![Documentation Status](https://readthedocs.org/projects/mmclassification/badge/?version=latest)](https://mmclassification.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/open-mmlab/mmclassification/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmclassification)
[![license](https://img.shields.io/github/license/open-mmlab/mmclassification.svg)](https://github.com/open-mmlab/mmclassification/blob/master/LICENSE)

## Introduction

English | [简体中文](/README_zh-CN.md)

MMClassification is an open source image classification toolbox based on PyTorch. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

Documentation: https://mmclassification.readthedocs.io/en/latest/

![demo](https://user-images.githubusercontent.com/9102141/87268895-3e0d0780-c4fe-11ea-849e-6140b7e0d4de.gif)

### Major features

- Various backbones and pretrained models
- Bag of training tricks
- Large-scale training configs
- High efficiency and extensibility

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.18.0 was released in 30/11/2021.

Highlights of the new version:
- Support **MLP-Mixer** backbone and provide pre-trained checkpoints.
- Add a tool to **visualize the learning rate curve** of the training phase. Welcome to use with the [tutorial](https://mmclassification.readthedocs.io/en/latest/tools/visualization.html#learning-rate-schedule-visualization)!

v0.17.0 was released in 29/10/2021.

Highlights of this version:
- Support **Tokens-to-Token ViT** backbone and **Res2Net** backbone. Welcome to use!
- Support **ImageNet21k** dataset.
- Add a **pipeline visualization** tool. Try it with the [tutorials](https://mmclassification.readthedocs.io/en/latest/tools/visualization.html#pipeline-visualization)!

Please refer to [changelog.md](docs/en/changelog.md) for more details and other release history.

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

Supported backbones:

- [x] VGG
- [x] ResNet
- [x] ResNeXt
- [x] SE-ResNet
- [x] SE-ResNeXt
- [x] RegNet
- [x] ShuffleNetV1
- [x] ShuffleNetV2
- [x] MobileNetV2
- [x] MobileNetV3
- [x] Swin-Transformer
- [x] RepVGG
- [x] Vision-Transformer
- [x] Transformer-in-Transformer
- [x] Res2Net
- [x] MLP-Mixer
- [ ] DeiT
- [ ] Conformer
- [ ] EfficientNet

## Installation

Please refer to [install.md](docs/en/install.md) for installation and dataset preparation.

## Getting Started
Please see [getting_started.md](docs/en/getting_started.md) for the basic usage of MMClassification. There are also tutorials:

- [learn about configs](docs/en/tutorials/config.md)
- [finetuning models](docs/en/tutorials/finetune.md)
- [adding new dataset](docs/en/tutorials/new_dataset.md)
- [designing data pipeline](docs/en/tutorials/data_pipeline.md)
- [adding new modules](docs/en/tutorials/new_modules.md)
- [customizing schedule](docs/en/tutorials/schedule.md)
- [customizing runtime settings](docs/en/tutorials/runtime.md)

Colab tutorials are also provided. To learn about MMClassification Python API, you may preview the notebook [here](https://github.com/open-mmlab/mmclassification/blob/master/docs/en/tutorials/MMClassification_python.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmclassification/blob/master/docs/en/tutorials/MMClassification_python.ipynb) on Colab.
To learn about MMClassification shell tools, you may preview the notebook [here](https://github.com/open-mmlab/mmclassification/blob/master/docs/en/tutorials/MMClassification_tools.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmclassification/blob/master/docs/en/tutorials/MMClassification_tools.ipynb) on Colab.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{2020mmclassification,
    title={OpenMMLab's Image Classification Toolbox and Benchmark},
    author={MMClassification Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmclassification}},
    year={2020}
}
```

## Contributing

We appreciate all contributions to improve MMClassification.
Please refer to [CONTRUBUTING.md](docs/en/community/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMClassification is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new classifiers.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab toolbox for text detection, recognition and understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMlab toolkit for generative models.
- [MMFlow](https://github.com/open-mmlab/mmflow) OpenMMLab optical flow toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab FewShot Learning Toolbox and Benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D Human Parametric Model Toolbox and Benchmark.

## 集成自己的数据集

#### 1. 准备数据集

首先准备好数据集，并搞成如下的文件结构：

```
imagenet
├── meta
|	├── train.txt
|	├── val.txt
├── train
│   ├──	bingpian
│   │   ├── 026.JPEG
│   │   ├── ...
│   ├── bubingpian
│   │   ├── 999.JPEG
│   │   ├── ...
│   ├── ...
├── val
│   ├── bingpian
│   │   ├── 0027.JPEG
│   │   ├── ...
│   ├── bubingpian
│   │   ├── 993.JPEG
│   │   ├── ...
│   ├── ...
```

其中bingpian和bupingpian为类别

train.txt和val.txt的格式如下(以train为例，val也是一样的)，用于导入到mmclassification中

```
bingpian/026.JPEG 0
bubingpian/999.JPEG 1
```

生成train.txt代码（val也一样）

```
import os

path1 = 'train/bingpian'
path2 = 'train/bubingpian'
f = open('train.txt', 'w')
for file in os.listdir(path1):
    f.write('bingpian/' + file + " " + '0' + '\n')
for file in os.listdir(path2):
    f.write('bubingpian/' + file + " " + '1' + '\n')
```

#### 2. 修改MMClassification代码

```
'mmcls/datasets'目录下新建py文件（名字自取，以'mydataset.py'为例），写入内容如下：(#对应自己的类别)
```

```
import mmcv
import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset

@DATASETS.register_module()     # 一定要加，用于注册
class MyDataset(BaseDataset):   # MyDataset注意大小写，之后调用的时候一定要一模一样
    CLASSES = ["bingpian","bubingpian"]   #对应自己的类别
    def load_annotations(self):
        assert isinstance(self.ann_file, str)
        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos
```

```
'mmcls/datasets'目录下修改'__init__.py'文件,添加内容如下：
```

注意：**'MyDataset'**与上面代码块中class **MyDataset**(BaseDataset)一致

```
from .base_dataset import BaseDataset
from .builder import (DATASETS, PIPELINES, SAMPLERS, build_dataloader,
                      build_dataset, build_sampler)
from .cifar import CIFAR10, CIFAR100
from .dataset_wrappers import (ClassBalancedDataset, ConcatDataset,
                               RepeatDataset)
from .imagenet import ImageNet
from .imagenet21k import ImageNet21k
from .mnist import MNIST, FashionMNIST
from .multi_label import MultiLabelDataset
from .samplers import DistributedSampler, RepeatAugSampler
from .voc import VOC
from .mydataset import MyDataset

__all__ = [
    'BaseDataset', 'ImageNet', 'CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST',
    'VOC', 'MultiLabelDataset', 'build_dataloader', 'build_dataset',
    'DistributedSampler', 'ConcatDataset', 'RepeatDataset',
    'ClassBalancedDataset', 'DATASETS', 'PIPELINES', 'ImageNet21k', 'SAMPLERS',
    'build_sampler', 'RepeatAugSampler', 'MyDataset'
    ]
```

#### 3. 修改Config文件

```
'configs/_base_/datasets'目录下新建'mydataset.py'文件，写入内容如下：(#修改为自己的路径)
```

```
dataset_type = 'MyDataset'#修改为自己的数据集
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='train_path',#修改为自己的路径
        ann_file='train.txt_path',#修改为自己的路径
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='val_path',#修改为自己的路径
        ann_file='val.txt_path',#修改为自己的路径*
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='test_path',#修改为自己的路径
        ann_file='test.txt_path',#修改为自己的路径
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')
```

```
'configs/mobilenetv2/mobilenet-v2_8xb32_mydataset.py'，修改为如下内容：
```

```
_base_ = [
    '../_base_/models/mobilenet_v2_mydataset.py',                       
    '../_base_/datasets/mydataset.py',   
    '../_base_/schedules/mydataset_bs16.py',
    '../_base_/default_runtime.py'                         
]
```

#### 踩坑：

1. 由于我的项目只有bingpian和bubingpian两类，因此我把'../_base_/models/mobilenet_v2_mydataset.py' 的topk=（1,5）改成了1，只显示top1的值。相应的，在XXX文件下也要修改两处Topk（记不清了，看报错信息很容易定位到，out of range之类的）
2. 一定要安装开发者模式，dev那个
3. 大小写分清楚，分不清楚就全小写！！

