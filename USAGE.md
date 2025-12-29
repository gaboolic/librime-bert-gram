# BERT Grammar 使用说明

## 功能说明

BERT Grammar 插件使用 BERT 模型来评估词条在给定上下文中的概率，从而提供更智能的输入建议。

## 配置步骤

### 1. 准备模型文件

你需要准备两个文件：
- **BERT 模型** (`.onnx` 格式)
- **词汇表文件** (`vocab.txt`)

#### 获取模型

**选项 A: 使用预训练模型**
```python
from transformers import BertForMaskedLM, BertTokenizer
import torch

# 加载中文 BERT 模型
model = BertForMaskedLM.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 导出为 ONNX
dummy_input = torch.randint(0, 21128, (1, 128))
torch.onnx.export(
    model,
    dummy_input,
    "bert_model.onnx",
    input_names=['input_ids'],
    output_names=['logits'],
    opset_version=11,
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'}
    }
)

# 保存词汇表
tokenizer.save_vocabulary("vocab.txt")
```

**选项 B: 使用量化模型（推荐，更快）**
```python
# 使用 onnxruntime 进行量化
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "bert_model.onnx",
    "bert_model_quantized.onnx",
    weight_type=QuantType.QUInt8
)
```

### 2. 放置文件

将模型文件放到 Rime 的共享数据目录：

```bash
# Linux/macOS
mkdir -p ~/.config/ibus-rime/bert_grammar
cp bert_model.onnx ~/.config/ibus-rime/bert_grammar/
cp vocab.txt ~/.config/ibus-rime/bert_grammar/

# Windows
# 通常是 %APPDATA%\Rime\bert_grammar\
```

### 3. 配置 Schema

在你的 schema 配置文件中启用上下文建议：

```yaml
# luna_pinyin.schema.yaml 或其他 schema 文件
translator:
  dictionary: luna_pinyin
  contextual_suggestions: true  # 启用上下文建议
```

在 `default.yaml` 或 schema 配置中添加 BERT Grammar 配置：

```yaml
# default.yaml
bert_grammar:
  model_path: "bert_grammar/bert_model.onnx"  # 相对路径
  vocab_path: "bert_grammar/vocab.txt"
```

或者使用绝对路径：

```yaml
bert_grammar:
  model_path: "/path/to/bert_model.onnx"
  vocab_path: "/path/to/vocab.txt"
```

### 4. 重新部署

```bash
rime_deployer --build
```

## 工作原理

1. **用户输入拼音**：例如输入 "nihao"
2. **字典查找**：从字典中查找候选词（"你好"、"你号" 等）
3. **获取上下文**：获取之前已输入的文字（例如 "今天"）
4. **BERT 评估**：使用 BERT 模型计算 `P(候选词 | 上下文)`
5. **权重调整**：`最终权重 = 字典权重 + log(P(候选词|上下文))`
6. **排序输出**：按最终权重排序，展示给用户

## 性能优化

### 使用量化模型

量化模型可以显著提升性能：

```python
# INT8 量化（推荐）
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "bert_model.onnx",
    "bert_model_int8.onnx",
    weight_type=QuantType.QUInt8
)
```

量化后的模型：
- **速度**：提升 2-3 倍
- **内存**：减少约 75%
- **精度**：略有下降，但通常可接受

### 模型选择

- **bert-base-chinese**: 平衡性能和精度
- **bert-small-chinese**: 更快，但精度略低
- **自定义模型**: 针对输入法场景优化的模型

## 故障排除

### 问题 1: 插件未加载

**检查日志**：
```bash
# Linux/macOS
tail -f ~/.config/ibus-rime/rime.ibus.log

# 查找 "registering BERT grammar component"
```

**解决方案**：
- 确保插件已编译
- 检查 `plugins/bert-grammar` 目录存在
- 确保 ONNX Runtime 库可用

### 问题 2: 模型加载失败

**检查**：
- 模型文件路径是否正确
- 模型文件是否存在
- 文件权限是否正确

**日志信息**：
```
BertGrammar: loading model from ...
BertGrammar: model loaded successfully
```

### 问题 3: 推理速度慢

**优化方法**：
1. 使用量化模型（INT8）
2. 使用更小的模型
3. 启用 GPU 加速（如果可用）
4. 添加缓存机制

### 问题 4: 结果不理想

**可能原因**：
1. 模型不适合输入法场景（使用通用 BERT）
2. Tokenizer 实现不完善
3. 概率提取逻辑需要调整

**改进方向**：
- 使用针对输入法训练的模型
- 改进 tokenizer 实现
- 调整概率计算逻辑

## 高级配置

### 自定义 Tokenizer

如果需要更精确的 tokenization，可以实现自定义 tokenizer：

```cpp
// 在 bert_grammar.cc 中改进 Tokenize 方法
// 使用 HuggingFace tokenizers C++ 库或实现完整的 WordPiece
```

### GPU 加速

如果系统有 NVIDIA GPU，可以启用 CUDA：

```cpp
// 在 bert_grammar.cc 的 LoadSession 中取消注释：
OrtCUDAProviderOptions cuda_options{};
session_options.AppendExecutionProvider_CUDA(cuda_options);
```

### 批处理

对于批量查询，可以实现批处理以提高效率。

## 注意事项

1. **内存占用**：BERT 模型较大，注意内存使用
2. **首次加载**：首次加载模型可能较慢
3. **兼容性**：确保 ONNX 模型版本与 ONNX Runtime 兼容
4. **性能**：BERT 推理比 n-gram 慢，但通常仍在可接受范围内

## 参考

- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)
- [BERT 模型转换指南](IMPLEMENTATION_GUIDE.md)
- [编译说明](BUILD_INSTRUCTIONS.md)


