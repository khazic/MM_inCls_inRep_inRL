# 入门指南

## 环境配置

### 硬件要求
- GPU: NVIDIA GPU with CUDA 12.1+
- 内存: 建议 32GB+
- 存储: 建议 100GB+ SSD

### 软件环境
1. 安装 CUDA 和 cuDNN
```bash
# 检查CUDA版本
nvidia-smi
```

2. 安装 Conda
```bash
# 下载并安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

3. 创建环境
```bash
conda create -n mmcls python=3.10
conda activate mmcls
```

4. 安装依赖
```bash
pip install -r requirements.txt
pip install -e .
```

## 数据准备

### 数据格式
训练数据需要准备成以下JSON格式：
```json
{
    "image": "path/to/image.jpg",
    "text": "图像描述文本",
    "label": [0, 1, 0, ...]  # 标签向量
}
```

### 数据目录结构
```
data/
├── cls_data/
│   ├── train_dataset.json  # 训练集
│   ├── val_dataset.json    # 验证集
│   └── label.json         # 标签映射文件
└── images/                # 图像文件目录
```

## 模型训练

### 单GPU训练
```bash
cd examples/classification
bash train_qwen2vl.sh
```

### 多GPU训练
修改 `train_qwen2vl.sh` 中的 `CUDA_VISIBLE_DEVICES` 和 `NUM_GPU` 参数：
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPU=4
```

### 训练参数说明
- `learning_rate`: 建议范围 1e-5 ~ 5e-5
- `num_train_epochs`: 建议 2-5 轮
- `per_device_train_batch_size`: 根据GPU显存调整
- `gradient_accumulation_steps`: 可用于增加等效批次大小

## 模型评估

使用验证集评估模型性能：
```bash
cd examples/classification
bash eval_qwen2vl.sh
```

## 常见问题

1. **OOM问题**
   - 减小batch size
   - 使用梯度累积
   - 启用 DeepSpeed

2. **训练不收敛**
   - 检查学习率
   - 检查数据预处理
   - 验证标签格式

3. **CUDA错误**
   - 确认CUDA版本匹配
   - 检查GPU显存使用
   - 清理GPU缓存

## 性能优化建议

1. **数据加载优化**
   - 使用 `num_workers`
   - 启用 `pin_memory`
   - 预缓存数据集

2. **训练加速**
   - 使用 Flash Attention
   - 启用 AMP 混合精度训练
   - 使用 xFormers 优化注意力计算

3. **内存优化**
   - 使用梯度检查点
   - 启用梯度累积
   - 使用 DeepSpeed ZeRO 优化 