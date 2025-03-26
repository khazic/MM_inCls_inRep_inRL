# MultiModal Classification and Representation Training

多模态分类与表征训练框架 (MultiModal Classification and Representation Training Framework)

## 项目简介

这是一个基于 PyTorch 的多模态分类和表征学习框架，专门用于处理视觉-语言任务。该框架支持使用最新的预训练模型（如 Qwen2-VL）进行多模态分类任务的训练和评估。

### 主要特性

- 🚀 支持多种视觉-语言预训练模型
- 📊 灵活的数据处理和数据增强pipeline
- 🔧 模块化设计，易于扩展
- 📈 内置评估指标和可视化工具
- 🎯 支持多GPU分布式训练

## 项目结构

```
MM_inCls_inRep_inRL/
├── src/                    # 源代码目录
│   ├── modeling/          # 模型定义
│   ├── tasks/             # 任务相关代码
│   ├── utils/             # 工具函数
│   ├── metrics/           # 评估指标
│   └── models/            # 模型架构
├── data/                  # 数据目录
├── examples/              # 示例代码
├── experiment/           # 实验输出目录
└── docs/                 # 文档
```

## 环境要求

- Python 3.10+
- PyTorch 2.4.0
- CUDA 12.1+

## 安装说明

1. 克隆仓库：
```bash
git clone [your-repo-url]
cd MM_inCls_inRep_inRL
```

2. 创建并激活虚拟环境：
```bash
conda create -n mmcls python=3.10
conda activate mmcls
```

3. 安装依赖：
```bash
pip install -r requirements.txt
pip install -e .
```

## 快速开始

### 训练模型

使用以下命令开始训练：

```bash
cd examples/classification
bash train_qwen2vl.sh
```

训练脚本支持以下主要参数：

- `--model_name_or_path`: 预训练模型路径
- `--train_file`: 训练数据文件
- `--validation_file`: 验证数据文件
- `--output_dir`: 输出目录
- `--learning_rate`: 学习率
- `--num_train_epochs`: 训练轮数

### 数据格式

训练数据应为JSON格式，包含以下字段：

```json
{
    "image": "图像路径",
    "text": "文本描述",
    "label": [标签列表]
}
```

## 主要组件

- **modeling_qwen2_vl_classification.py**: Qwen2-VL模型的分类任务适配
- **tool.py**: 数据处理和辅助工具
- **metrics/f1.py**: F1评估指标实现

## 性能优化

- 使用 Flash Attention 加速注意力计算
- 支持 DeepSpeed 进行分布式训练
- 使用 xFormers 优化内存使用

## 引用

如果您使用了本项目，请引用：

```bibtex
@misc{MM_inCls_inRep_inRL,
    title={MultiModal Classification and Representation Training Framework},
    author={Your Name},
    year={2024},
    publisher={GitHub},
    howpublished={\url{https://github.com/yourusername/MM_inCls_inRep_inRL}}
}
```

## 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题，请通过 Issues 或以下方式联系我们：

- 邮箱：[khazzz1c@gmail.com]
