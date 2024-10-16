```
GAN_Project/
│
├── common/                       # 通用模块
│   ├── __init__.py
│   ├── layers.py                 # 通用神经网络层（比如自定义层，SAGAN中的Self-Attention层）
│   ├── loss_functions.py         # 通用损失函数（如对抗损失，L2损失等）
│   ├── optimizers.py             # 通用优化器（SGD, Adam等）
│   └── utils.py                  # 一些常用工具函数（数据加载器、图像处理、可视化等）
│
├── core/                         # 核心模块（算法和模型实现）
│   ├── __init__.py
│   ├── base_gan.py               # GAN基类，定义通用的属性和方法
│   ├── vanilla_gan.py            # 原始GAN的实现
│   ├── sagan.py                  # Self-Attention GAN的实现
│   ├── biggan.py                 # BigGAN的实现
│   └── gan_factory.py            # 工厂模式，实例化不同的GAN模型
│
├── config/                       # 配置和超参数管理
│   ├── __init__.py
│   ├── gan_config.yaml           # GAN相关的超参数（batch size, learning rate等）
│   └── model_config.yaml         # 模型相关的配置（生成器、判别器架构等）
│
├── data/                         # 数据相关
│   ├── __init__.py
│   ├── data_loader.py            # 数据加载类，处理数据集的预处理、批次化等
│   ├── datasets/                 # 数据集存放目录
│   │   └── mnist/                # 示例数据集（MNIST、CIFAR-10等）
│   └── transforms.py             # 数据增强和转换（裁剪、归一化等）
│
├── experiments/                  # 实验管理，保存不同实验的超参数和模型
│   ├── __init__.py
│   ├── exp_01/                   # 实验1目录
│   │   ├── model.pth             # 训练的模型文件
│   │   ├── params.yaml           # 本次实验的超参数设置
│   │   └── logs/                 # 日志文件、训练过程中的输出
│   ├── exp_02/                   # 实验2目录
│   └── ...
│
├── output/                       # 生成图像或模型输出文件
│   ├── generated_images/         # 存放生成的图像文件
│   ├── results.csv               # 记录训练和测试的指标
│   └── model_weights/            # 训练好的模型权重文件
│
├── tests/                        # 单元测试目录
│   ├── __init__.py
│   ├── test_gan.py               # 测试GAN模型的训练和推理功能
│   └── test_utils.py             # 测试工具函数
│
├── main.py                       # 训练和评估的主入口
├── README.md                     # 项目简介
└── requirements.txt              # 项目依赖的Python库

```

