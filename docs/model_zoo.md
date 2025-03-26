# 模型库

## 预训练模型

### Qwen2-VL

#### 模型信息
- **名称**: Qwen2-VL-2B-Instruct
- **参数量**: 2B
- **架构**: Transformer-based
- **训练数据**: 多模态指令数据
- **许可证**: 商业许可

#### 使用方法
```python
from transformers import AutoProcessor, AutoModel
from src.modeling.modeling_qwen2_vl_classification import Qwen2VLForClassification

# 加载处理器
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# 加载模型
model = Qwen2VLForClassification.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    num_labels=num_classes
)
```

#### 性能指标
- **图文分类准确率**: 85%+
- **推理速度**: 200样本/秒 (单个V100)
- **显存占用**: 8GB (batch_size=32)

### 其他支持的模型

1. **LLaVA**
   - 参数量: 7B/13B
   - 特点: 强大的视觉-语言理解能力

2. **MiniCPM-V**
   - 参数量: 1B
   - 特点: 轻量级多模态模型

3. **CogVLM**
   - 参数量: 17B
   - 特点: 认知级视觉语言模型

## 微调模型

### 分类模型

1. **通用图文分类**
   ```python
   model = Qwen2VLForClassification.from_pretrained(
       "path/to/finetuned/model",
       num_labels=1000
   )
   ```

2. **多标签分类**
   ```python
   model = Qwen2VLForClassification.from_pretrained(
       "path/to/finetuned/model",
       problem_type="multi_label_classification",
       num_labels=100
   )
   ```

### 表征模型

1. **特征提取器**
   ```python
   model = Qwen2VLForClassification.from_pretrained(
       "path/to/finetuned/model",
       output_hidden_states=True
   )
   ```

## 模型配置

### 训练配置

1. **单GPU配置**
```json
{
    "learning_rate": 1e-5,
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 3
}
```

2. **多GPU配置**
```json
{
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 3,
    "local_rank": -1,
    "deepspeed": "ds_config.json"
}
```

### DeepSpeed配置

```json
{
    "train_micro_batch_size_per_gpu": 16,
    "gradient_accumulation_steps": 2,
    "steps_per_print": 100,
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "reduce_scatter": true,
        "overlap_comm": true
    }
}
```

## 模型评估

### 评估指标

1. **分类指标**
   - 准确率 (Accuracy)
   - F1分数
   - 精确率 (Precision)
   - 召回率 (Recall)

2. **表征指标**
   - 余弦相似度
   - 特征对齐度

### 评估脚本

```bash
python eval.py \
    --model_name_or_path path/to/model \
    --eval_data path/to/eval.json \
    --batch_size 32 \
    --metric accuracy,f1
```

## 模型部署

### TorchScript导出

```python
model.eval()
traced_model = torch.jit.trace(model, example_inputs)
torch.jit.save(traced_model, "model.pt")
```

### ONNX导出

```python
torch.onnx.export(
    model,
    example_inputs,
    "model.onnx",
    opset_version=12
)
```

### TensorRT优化

```bash
trtexec --onnx=model.onnx \
        --saveEngine=model.trt \
        --fp16 \
        --workspace=4096
```

## 最佳实践

1. **模型选择**
   - 小数据集: Qwen2-VL-2B
   - 大数据集: Qwen2-VL-7B

2. **训练策略**
   - 学习率预热
   - 梯度裁剪
   - 混合精度训练

3. **推理优化**
   - 批处理推理
   - 模型量化
   - 动态批处理
``` 