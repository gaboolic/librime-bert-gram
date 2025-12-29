# BERT Grammar 测试指南

本文档说明如何测试 BERT Grammar 组件是否正常工作。

## 方法 1: 使用 C++ 测试程序（推荐）

### 编译测试程序

1. **配置 CMake 时启用测试程序**：

**Linux/macOS**:
```bash
mkdir build
cd build
cmake .. -DBUILD_TEST=ON
```

**Windows**:
```cmd
mkdir build
cd build
cmake .. -DBUILD_TEST=ON -DRIME_ROOT_DIR="C:/path/to/rime"
```

如果 CMake 找不到 Rime 库，可以手动指定路径：
```bash
cmake .. -DBUILD_TEST=ON -DRIME_ROOT_DIR=/path/to/rime
```

2. **编译**：
```bash
cmake --build .
# 或 Windows 上使用 Visual Studio:
cmake --build . --config Release
```

3. **运行测试**：
```bash
# 基本用法
./test_bert_grammar <model_path> <vocab_path>

# 示例
./test_bert_grammar ../bert_grammar/model.onnx ../bert_grammar/vocab.txt

# 如果使用配置文件
./test_bert_grammar model.onnx vocab.txt /path/to/default.yaml
```

### 测试程序功能

测试程序会执行以下测试：

1. **模型加载测试**
   - 检查模型是否成功加载
   - 验证 ONNX Runtime 会话是否正常

2. **词汇表测试**
   - 测试常见中文词汇的识别
   - 验证词汇表是否正确加载

3. **上下文相关查询测试**
   - 测试不同上下文下的词条评分
   - 验证 Query 方法是否正常工作

4. **上下文敏感性测试**
   - 验证相同词条在不同上下文下是否产生不同评分
   - 这是验证 BERT 模型是否真正生效的关键测试

5. **性能测试**
   - 测量查询延迟
   - 计算吞吐量

### 预期输出

如果模型正常工作，你应该看到：

```
============================================================
测试 1: 模型加载状态
============================================================
✓ 模型似乎已加载（返回值不是默认值）
   返回值: -2.345678

============================================================
测试 4: 上下文敏感性
============================================================
✓ 不同上下文产生了不同的评分（模型正常工作）
```

### 故障排除

**问题 1: 模型加载失败**

如果看到：
```
⚠ 警告: 返回值为默认惩罚值，模型可能未正确加载
```

检查：
- 模型文件路径是否正确
- 模型文件是否存在
- ONNX Runtime 是否正确链接
- 查看日志中的错误信息

**问题 2: 所有上下文产生相同评分**

如果看到：
```
⚠ 警告: 不同上下文产生了相似的评分
```

可能原因：
- 模型输出处理逻辑有问题
- Tokenizer 实现不正确
- 模型本身的问题

**问题 3: 编译错误**

如果编译失败，检查：
- 是否安装了 glog
- Rime 开发库是否正确安装
- ONNX Runtime 是否正确配置

**Windows 特定问题**：

如果遇到 "Could NOT find PkgConfig" 错误：
- 这是正常的，Windows 不使用 PkgConfig
- 确保已安装 Rime 开发库
- 使用 `-DRIME_ROOT_DIR` 指定 Rime 安装路径：
  ```cmd
  cmake .. -DBUILD_TEST=ON -DRIME_ROOT_DIR="C:/path/to/rime"
  ```

如果找不到 Rime 库：
- 检查 Rime 是否已正确安装
- 尝试使用 `find_package(rime)` 或设置 `RIME_ROOT_DIR`
- 确保 Rime 的 include 和 lib 目录结构正确

## 方法 2: 查看日志输出

### 在 Rime 中使用时查看日志

1. **Linux/macOS**：
```bash
tail -f ~/.config/ibus-rime/rime.ibus.log
# 或
tail -f ~/.local/share/fcitx5/rime/rime.log
```

2. **Windows**：
查看 `%APPDATA%\Rime\rime.log`

### 关键日志信息

查找以下日志信息来确认模型是否生效：

**成功加载模型**：
```
I[日期时间] BertGrammar: model loaded successfully
I[日期时间] ONNX Runtime session loaded successfully
I[日期时间] Loaded vocabulary with X tokens
```

**模型推理**：
```
I[日期时间] BERT inference completed
```

**错误信息**：
```
E[日期时间] Failed to load ONNX model: ...
E[日期时间] BERT inference failed: ...
```

## 方法 3: 实际使用测试

### 配置 Rime 使用 BERT Grammar

1. **在 `default.yaml` 中配置**：
```yaml
bert_grammar:
  model_path: "bert_grammar/model.onnx"
  vocab_path: "bert_grammar/vocab.txt"
```

2. **在 schema 中启用上下文建议**：
```yaml
translator:
  dictionary: luna_pinyin
  contextual_suggestions: true
```

3. **重新部署**：
```bash
rime_deployer --build
```

### 测试场景

1. **输入上下文相关的句子**：
   - 输入 "今天天气很好"
   - 继续输入 "我们"
   - 观察候选词排序是否考虑了上下文

2. **对比测试**：
   - 输入 "我喜欢吃" 然后输入 "苹果"
   - 输入 "我在看" 然后输入 "苹果"
   - 观察 "苹果" 的排序是否不同

3. **长上下文测试**：
   - 输入 "各个国家有各个国家的" 然后输入 "国歌"
   - 观察是否正确识别为 "国歌" 而不是其他候选

## 方法 4: Python 测试脚本

项目中的 Python 测试脚本也可以用来验证模型：

```bash
# 测试 ONNX 模型
python python/test_onnx_model.py

# 测试评分功能
python python/simple_example.py
```

## 验证清单

- [ ] 测试程序编译成功
- [ ] 模型加载测试通过
- [ ] 词汇表测试通过
- [ ] 上下文查询测试通过
- [ ] 上下文敏感性测试通过（关键！）
- [ ] 日志中看到 "model loaded successfully"
- [ ] 实际使用中候选词排序考虑了上下文
- [ ] 性能在可接受范围内（< 500ms/查询）

## 常见问题

**Q: 如何确认 BERT 模型真的在工作？**

A: 最可靠的方法是运行上下文敏感性测试。如果相同词条在不同上下文下产生不同评分，说明模型在正常工作。

**Q: 测试程序显示模型已加载，但实际使用中没有效果？**

A: 检查：
1. Rime 配置中是否正确启用了 `contextual_suggestions`
2. 是否重新部署了配置
3. 查看 Rime 日志确认组件是否被加载

**Q: 性能太慢怎么办？**

A: 
1. 使用量化模型（INT8）
2. 使用更小的模型
3. 启用 GPU 加速（如果可用）
4. 检查是否有缓存机制

**Q: 如何调试推理过程？**

A: 
1. 启用详细日志：设置 `FLAGS_v=2` 或更高
2. 在代码中添加更多 LOG 语句
3. 使用 Python 脚本对比输出

## 性能基准

参考性能指标（CPU，未量化模型）：
- 单次查询：50-200ms
- 吞吐量：5-20 查询/秒

参考性能指标（CPU，INT8 量化模型）：
- 单次查询：20-100ms
- 吞吐量：10-50 查询/秒

如果性能远低于这些指标，考虑优化或使用量化模型。

