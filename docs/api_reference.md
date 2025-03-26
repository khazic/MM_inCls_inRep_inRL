# API 参考文档

## 模型 API

### Qwen2VLForClassification

```python
from src.modeling.modeling_qwen2_vl_classification import Qwen2VLForClassification

model = Qwen2VLForClassification.from_pretrained(
    pretrained_model_name_or_path,
    num_labels=num_labels
)
```

#### 参数
- `pretrained_model_name_or_path` (str): 预训练模型路径
- `num_labels` (int): 分类标签数量
- `problem_type` (str, optional): 问题类型，可选 'single_label_classification' 或 'multi_label_classification'

#### 方法
- `forward()`: 前向传播
- `prepare_inputs_for_generation()`: 准备生成输入

## 数据处理 API

### VLClassificationDataCollatorWithPadding

```python
from src.utils.tool import VLClassificationDataCollatorWithPadding

data_collator = VLClassificationDataCollatorWithPadding(
    processor=processor,
    padding=True
)
```

#### 参数
- `processor`: 模型处理器
- `padding` (bool): 是否进行填充
- `max_length` (int, optional): 最大序列长度

### PreprocessDataset

```python
from src.utils.tool import PreprocessDataset

dataset = PreprocessDataset(
    dataset,
    processor=processor
)
```

#### 参数
- `dataset`: 原始数据集
- `processor`: 模型处理器
- `max_length` (int, optional): 最大序列长度

## 评估指标 API

### F1评分

```python
from src.metrics.f1 import F1

metric = F1()
results = metric.compute(predictions=preds, references=labels)
```

#### 参数
- `predictions`: 模型预测结果
- `references`: 真实标签
- `average` (str, optional): 平均方式，可选 'micro', 'macro', 'weighted'

## 训练器 API

### 自定义训练器

```python
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        自定义损失计算
        """
        pass

    def training_step(self, model, inputs):
        """
        自定义训练步骤
        """
        pass
```

## 工具函数

### mm_preprocess

```python
from src.utils.tool import mm_preprocess

processed_data = mm_preprocess(
    images,
    text,
    processor
)
```

#### 参数
- `images`: 图像数据
- `text`: 文本数据
- `processor`: 模型处理器

## 配置类

### ModelArguments

```python
@dataclass
class ModelArguments:
    model_name_or_path: str
    cache_dir: Optional[str] = None
    model_revision: str = "main"
    use_auth_token: bool = False
```

### DataTrainingArguments

```python
@dataclass
class DataTrainingArguments:
    train_file: str
    validation_file: Optional[str] = None
    max_seq_length: int = 512
    pad_to_max_length: bool = False
```

## 环境变量

主要的环境变量配置：

```bash
# CUDA设备
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 分布式训练
export MASTER_PORT=29500
export MASTER_ADDR=localhost

# Wandb配置
export WANDB_PROJECT=mm_classification
export WANDB_WATCH=false

# DeepSpeed配置
export DS_ACCELERATOR=cuda
```

## 实用工具

### 模型保存与加载

```python
# 保存模型
model.save_pretrained("path/to/save")

# 加载模型
model = Qwen2VLForClassification.from_pretrained("path/to/load")
```

### 分布式训练

```python
# 初始化分布式环境
torch.distributed.init_process_group(backend="nccl")

# 设置设备
model = model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model)
```

### 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(**inputs)
    loss = outputs.loss
``` 