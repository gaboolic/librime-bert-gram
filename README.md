# librime-bert-grammar

使用 BERT 模型替代 n-gram 模型（如 octagram）的 Rime 插件。
本项目处于开发阶段

## 功能

这个插件提供了一个基于 BERT 的 Grammar 组件，用于：
- 上下文相关的词条权重评估
- 更准确的句子生成
- 更好的输入法智能推荐

## 编译要求

1. librime 开发环境
2. BERT 推理库（选择其一）：
   - ONNX Runtime
   - PyTorch C++ API
   - TensorFlow C++ API
   - 或其他 BERT 推理库

## 配置

在 schema 配置文件中添加：

```yaml
grammar:
  model_path: "bert_grammar/model.onnx"  # 相对路径或绝对路径
  vocab_path: "bert_grammar/vocab.txt"   # BERT tokenizer 词汇表
```

## 实现说明

### 1. 集成 BERT 推理库

在 `bert_grammar.cc` 的 `LoadModel()` 和 `ComputeProbability()` 方法中，你需要：

1. **选择 BERT 推理库**：
   - ONNX Runtime（推荐，跨平台，性能好）
   - PyTorch C++ API
   - TensorFlow C++ API
   - 其他支持 C++ 的推理库

2. **实现模型加载**：
   ```cpp
   bool BertGrammar::LoadModel() {
     // 使用你选择的库加载 BERT 模型
     // 例如 ONNX Runtime:
     // Ort::Session session(env, model_path_.c_str(), ...);
     return true;
   }
   ```

3. **实现概率计算**：
   ```cpp
   double BertGrammar::ComputeProbability(const string& context,
                                          const string& word) {
     // 1. Tokenize: 使用 BERT tokenizer 将 context 和 word 转换为 token IDs
     // 2. 构建输入: [CLS] context_tokens [SEP] word_tokens [SEP]
     // 3. 运行推理: 调用 BERT 模型
     // 4. 提取概率: 从输出中获取 P(word|context)
     // 5. 返回概率值
   }
   ```

### 2. BERT 模型输入格式

BERT 模型需要以下输入：
- **Token IDs**: 使用 BERT tokenizer 将文本转换为 token ID 序列
- **Attention Mask**: 指示哪些位置是真实 token，哪些是 padding
- **Token Type IDs**: 区分 context 和 word 部分（可选）

### 3. BERT 模型输出处理

BERT 模型的输出通常是：
- **CLS token 的 embedding**: 用于分类任务
- **所有 token 的 embeddings**: 用于序列标注

对于语言模型任务，你需要：
1. 使用 MLM（Masked Language Model）头获取概率
2. 或者使用序列到序列的方式计算 P(word|context)

### 4. 性能优化

- **批处理**: 如果可能，批量处理多个查询
- **模型量化**: 使用 INT8 量化模型减少内存和加速
- **缓存**: 缓存常见 context-word 对的概率
- **异步推理**: 使用异步推理避免阻塞

## 使用示例

### 在 schema.yaml 中启用

```yaml
translator:
  dictionary: luna_pinyin
  contextual_suggestions: true  # 启用上下文建议
  # BERT grammar 会自动被使用（如果已注册）
```

### 配置模型路径

```yaml
# default.yaml 或 schema.yaml
bert_grammar:
  model_path: "/path/to/bert_model.onnx"
  vocab_path: "/path/to/vocab.txt"
```

## 测试

如何验证 BERT 模型是否正常工作？请查看 [TESTING.md](TESTING.md) 获取详细的测试指南。

**快速测试**：
```bash
# 编译测试程序
mkdir build && cd build
cmake .. -DBUILD_TEST=ON
cmake --build .

# 运行测试
./test_bert_grammar <model_path> <vocab_path>
```

## 注意事项

1. **注册优先级**: 这个插件注册为 "grammar"，会覆盖 octagram 的注册。如果两个插件都加载，后加载的会生效。

2. **模型格式**: 确保 BERT 模型格式与你的推理库兼容（ONNX, TorchScript, TensorFlow SavedModel 等）。

3. **性能**: BERT 模型推理比 n-gram 慢，考虑使用量化模型或更小的模型。

4. **内存**: BERT 模型通常较大，注意内存使用。

5. **测试**: 使用测试程序验证模型是否正常工作，特别是上下文敏感性测试。
