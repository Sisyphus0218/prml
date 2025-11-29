# 模式识别与机器学习实践作业

采用 Logistic Regression、决策树和卷积神经网络，对 CIFAR-10 数据集进行图像分类

## 环境配置

```bash
conda create --name prml python=3.10
conda activate prml

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia

conda install scikit-learn
conda install matplotlib
conda install tensorboard
```

如果后续运行代码时报错 `undefined symbol: iJIT_NotifyEvent`

```bash
pip install mkl==2024.0.0
```

## 准备数据

将 `cifar-10-batches-py` 放在 `data` 文件夹下

## 运行代码

进入相应文件夹

```bash
cd logistic_regression
cd decision_tree
cd convolution_network
```

训练

```bash
python code/train.py
tensorboard --logdir=log
```

测试

```bash
python code/test.py
```

