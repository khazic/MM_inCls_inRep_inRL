# Multimodal_Search

基于大规模多模态模型的推荐搜索系统。

## 项目结构

```
Multimodal_Search/
├── multimodal_search/
│   ├── modeling/         # 模型相关代码
│   ├── data/            # 数据处理代码
│   ├── utils/           # 工具函数
│   └── evaluation/      # 评估指标
├── examples/            # 使用示例
└── tests/              # 测试代码
```

## 安装

```bash
git clone https://github.com/yourusername/Multimodal_Search.git
cd Multimodal_Search
pip install -e .
```

## 使用方法

### 训练

```bash
cd examples
bash train.sh
```

### 数据格式

训练数据需要包含以下字段:
- message: 输入消息
- label: 标签信息
- image/video: 图像或视频数据

## License

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。
