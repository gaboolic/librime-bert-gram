# BERT 使用指南

本项目演示如何使用BERT模型进行文本编码和相似度计算。

## 简介

Transformer 是一种神经网络架构，由Google在2017年提出，其核心创新是自注意力机制
BERT（Bidirectional Encoder Representations from Transformers）是基于Transformer架构的具体模型，由Google在2018年提出。BERT仅使用Transformer的编码器部分，通过堆叠多层编码器实现双向语义理解，专注于自然语言理解任务
GPT模型的主要结构是一个多层的Transformer解码器，但它只使用了Transformer解码器的部分，GPT演进了三个版本，gpt1 gpt2 gpt3，再后来演进到gpt3.5 就是chatgpt
简单来说，BERT和GPT是一体两面的关系。BERT和ngrams这种纯粹基于概率的模型不同，BERT是真的有一定的语义理解能力。

## 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装

## 快速开始

### 基本使用

```python
from transformers import BertTokenizer, BertModel

# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
model = BertModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

# 对文本进行编码
text = "这是一个示例文本"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# 获取句子表示（使用[CLS]标记）
sentence_embedding = outputs.last_hidden_state[:, 0, :]
```

### 运行示例

```bash
python example_usage.py
```

## 可用的BERT模型

Hugging Face上提供了多个预训练的BERT模型：

- `https://huggingface.co/google-bert/bert-base-chinese` - 通用中文模型

- 其他特定任务的模型

## 主要功能

### 1. 输入法流畅度评分（推荐）⭐

使用BERT评估句子的语法正确性和流畅度，适用于输入法候选排序：

```python
from input_method_scorer_v2 import InputMethodScorer

# 初始化评分器（推荐使用中文BERT，准确度高）
scorer = InputMethodScorer(model_name='bert-base-chinese', use_mlm_model=True)

# 比较两个句子
sentence1 = "各个国家有各个国家的国歌"  # 正确
sentence2 = "各个国家有各个国家德国个"  # 错误

# 使用综合分数（推荐方法，结合MLM和连贯性）
score1 = scorer.calculate_combined_score(sentence1)
score2 = scorer.calculate_combined_score(sentence2)

# 分数越高表示越流畅
print(f"句子1分数: {score1:.4f}")
print(f"句子2分数: {score2:.4f}")

# 对多个候选句子排序
candidates = [sentence1, sentence2]
ranked = scorer.rank_candidates(candidates, method='combined')
```

**模型选择：**
- `bert-base-chinese`（推荐）：中文BERT模型，准确度高，模型约400MB
- `huawei-noah/TinyBERT_General_4L_312D`：TinyBERT模型，模型小但准确度较低

**评分方法：**
- `combined`（推荐）：综合MLM和连贯性分数
- `mlm`：仅使用掩码语言模型分数
- `perplexity`：使用困惑度（越低越流畅）
- `coherence`：仅使用句子连贯性

运行示例：
```bash
python quick_start.py          # 快速开始（推荐）
python simple_example.py       # 详细示例
python input_method_scorer_v2.py  # 完整示例
```

### 2. 文本编码

将文本转换为向量表示：

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
model = BertModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

text = "这是一个示例文本"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
sentence_embedding = outputs.last_hidden_state[:, 0, :]
```

### 3. 句子相似度

计算两个句子的相似度：

```python
# 见 example_usage.py
python example_usage.py
```

## 测试结果

使用中文BERT模型 (`bert-base-chinese`) 的测试结果：

```
句子1: 各个国家有各个国家的国歌（正确）
  综合分数: 0.5243, 困惑度: 3.7875

句子2: 各个国家有各个国家德国个（错误）
  综合分数: 0.4370, 困惑度: 13.4481

✓ 所有指标都正确识别出句子1更流畅
```

## ONNX模型转换（用于C语言调用）

### 4. 将BERT模型转换为ONNX格式 ⭐

将BERT模型转换为ONNX格式，以便在C语言或其他语言中调用：

#### 基本转换

```bash
# 转换bert-base-chinese模型为ONNX格式
python convert_to_onnx.py
```

转换完成后，会在 `onnx_models/` 目录生成：
- `bert-base-chinese.onnx` - ONNX模型文件（可在C语言中调用）
- `tokenizer_config.json` - 分词器配置
- `vocab.txt` - 词汇表文件

#### 高级选项

```bash
# 指定模型名称
python convert_to_onnx.py --model bert-base-chinese

# 指定输出目录
python convert_to_onnx.py --output my_onnx_models

# 只转换编码器部分（更轻量，适合只需要句子嵌入的场景）
python convert_to_onnx.py --encoder-only

# 固定输入长度（不使用动态轴）
python convert_to_onnx.py --no-dynamic

# 指定ONNX opset版本
python convert_to_onnx.py --opset 13
```

#### 测试转换后的ONNX模型

```bash
# 基本测试
python test_onnx_model.py

# 与PyTorch模型比较输出（验证转换正确性）
python test_onnx_model.py --compare

# 指定模型和分词器路径
python test_onnx_model.py --model onnx_models/bert-base-chinese.onnx --tokenizer onnx_models
```

#### 转换脚本功能说明

**convert_to_onnx.py** 提供以下功能：
- 自动下载并加载BERT模型
- 转换为ONNX格式（支持动态输入长度）
- 自动保存分词器配置文件
- 验证转换后的ONNX模型
- 支持完整模型（MLM）或仅编码器转换

**test_onnx_model.py** 提供以下功能：
- 验证ONNX模型是否可以正常加载
- 测试模型推理功能
- 显示模型输入输出信息
- 可选：与PyTorch模型对比输出（验证转换正确性）

#### 在C语言中使用ONNX模型

转换后的ONNX模型可以在C语言中使用ONNX Runtime C API调用。详细说明请参考：
- `ONNX_C_USAGE.md` - 完整的C语言调用指南和示例代码

**重要提示：**
- 首次转换会下载模型（约400MB），需要网络连接
- 转换后的ONNX模型支持动态输入长度
- 在C语言中使用时，需要实现分词逻辑（将文本转换为input_ids）
- 需要下载并配置ONNX Runtime C库

## 注意事项

- 首次运行时会自动下载模型，需要网络连接
- `bert-base-chinese` 模型约400MB，下载需要一些时间
- `TinyBERT` 模型较小，但准确度较低
- 建议使用GPU加速推理（如果可用）
- 推荐使用 `bert-base-chinese` + `combined` 方法获得最佳效果

## 参考资源

- [TinyBERT论文](https://arxiv.org/abs/1909.10351)
- [Hugging Face模型库](https://huggingface.co/huawei-noah)

