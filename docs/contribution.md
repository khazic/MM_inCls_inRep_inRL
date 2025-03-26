# 贡献指南

感谢您对本项目的关注！我们欢迎各种形式的贡献，包括但不限于：

- 🐛 报告问题
- 💡 提出新功能
- 📝 改进文档
- 🔧 提交代码

## 开发流程

1. Fork 项目
2. 创建特性分支
3. 提交变更
4. 发起 Pull Request

## 代码规范

### Python 代码风格

- 遵循 PEP 8 规范
- 使用 4 个空格缩进
- 行长度限制在 120 字符以内
- 使用类型注解

示例：
```python
from typing import List, Optional

def process_data(
    images: List[str],
    text: str,
    labels: Optional[List[int]] = None
) -> dict:
    """
    处理输入数据。

    Args:
        images: 图像路径列表
        text: 文本描述
        labels: 可选的标签列表

    Returns:
        处理后的数据字典
    """
    pass
```

### 文档规范

- 所有公共函数、类必须有文档字符串
- 使用中文编写文档
- README 和文档使用 Markdown 格式

### 提交信息规范

格式：`<type>(<scope>): <subject>`

类型（type）：
- feat: 新功能
- fix: 修复
- docs: 文档
- style: 格式
- refactor: 重构
- test: 测试
- chore: 构建过程或辅助工具的变动

示例：
```
feat(model): 添加新的预训练模型支持
fix(trainer): 修复多GPU训练时的数据加载问题
docs(api): 更新API文档
```

## 测试规范

### 单元测试

使用 pytest 进行测试：

```python
def test_data_processor():
    processor = DataProcessor()
    result = processor.process(...)
    assert result["success"] == True
```

运行测试：
```bash
pytest tests/
```

### 集成测试

```bash
bash tests/integration/run_tests.sh
```

## 开发环境设置

1. 克隆项目：
```bash
git clone https://github.com/yourusername/MM_inCls_inRep_inRL.git
cd MM_inCls_inRep_inRL
```

2. 创建虚拟环境：
```bash
conda create -n mmcls python=3.10
conda activate mmcls
```

3. 安装开发依赖：
```bash
pip install -r requirements-dev.txt
```

4. 安装pre-commit钩子：
```bash
pre-commit install
```

## 分支管理

- main: 主分支，保持稳定
- develop: 开发分支
- feature/*: 特性分支
- bugfix/*: 修复分支
- release/*: 发布分支

## 发布流程

1. 版本号规范：遵循语义化版本
   - 主版本号：不兼容的API修改
   - 次版本号：向下兼容的功能性新增
   - 修订号：向下兼容的问题修正

2. 发布检查清单：
   - 更新版本号
   - 更新CHANGELOG.md
   - 更新文档
   - 运行测试套件
   - 创建发布标签

3. 发布命令：
```bash
# 更新版本号
bump2version patch  # 或 minor 或 major

# 创建标签
git tag v1.0.0
git push origin v1.0.0
```

## 问题反馈

1. 使用 GitHub Issues 提交问题
2. 提供详细的问题描述
3. 包含复现步骤
4. 附上相关日志和截图

## 联系方式

- 邮箱：[khazzz1c@gmail.com]
- 讨论组：[discussion-forum-link]

## 行为准则

1. 尊重所有贡献者
2. 保持专业和友善
3. 接受建设性批评
4. 关注问题本身

## 许可证

贡献的代码将采用项目的 Apache 2.0 许可证。 