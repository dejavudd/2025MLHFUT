# 2025MLHFUT
2025机器学习大作业合肥工业大学（复现顶会论文）

本次大作业我们所复现的是cvpr 2024的文章DAI-Net(https://github.com/ZPDu/DAI-Net) ，以及后续我们加入了IAT论文(https://github.com/cuiziteng/Illumination-Adaptive-Transformer) ，将这两篇论文的模型进行了融合优化。

# INSTALLATION

一开始我们需要克隆相关库，并创建对应的conda环境：
```
git clone https://github.com/ZPDu/DAI-Net.git
cd DAI-Net

conda create -y -n dainet python=3.7
conda activate dainet

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

在这里的话我的requirement.txt同时也满足与原来IAT的环境。

#TRAINING 

数据与权重准备
- 下载Widerface的训练集和测试集的图像 [WIDER FACE](http://shuoyang1213.me/WIDERFACE/).
- 获得训练集的标注 (https://github.com/daooshee/HLA-Face-Code/blob/main/train_code/dataset/wider_face_train.txt) 和验证集的标注(https://github.com/daooshee/HLA-Face-Code/blob/main/train_code/dataset/wider_face_val.txt).
- 下载Retinex Decomposition Net预训练权重 (https://drive.google.com/file/d/1MaRK-VZmjBvkm79E1G77vFccb_9GWrfG/view?usp=drive_link).
- 准备基础网络的预训练权重(https://drive.google.com/file/d/1whV71K42YYduOPjTTljBL8CB-Qs4Np6U/view?usp=drive_link) .

请将文件夹按下面的方式进行组织：

```
.
├── utils
├── weights
│   ├── decomp.pth
│   ├── vgg16_reducedfc.pth
├── dataset
│   ├── wider_face_train.txt
│   ├── wider_face_val.txt
│   ├── WiderFace
│   │   ├── WIDER_train
│   │   └── WIDER_val
```

#MODEL TRAINING

要训练模型的话，请你执行：
```
python -m torch.distributed.launch --nproc_per_node=$NUM_OF_GPUS$ train.py
```
其中可以自主选择要训练的GPU数目，项目会根据服务器中的已有gpu数目来优先选择空余的gpu进行对项目的训练，同时gpu的选择数目也会影响训练时候的学习率。
在项目里面的权重文件可以进行替换，从而实现通过DAI-Net的代码框架实现暗图像和高曝光图像的有效处理，在训练前需要再train.py代码中进行权重文件的路径替换。
