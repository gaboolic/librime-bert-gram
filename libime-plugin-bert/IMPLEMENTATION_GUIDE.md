# BERT 语言模型插件实现指南

## 概述

这个插件允许在不修改 <libime>(https://github.com/fcitx/libime) 源代码的情况下，将 ngram 语言模型替换为 BERT 模型。核心思路是创建一个继承自 `UserLanguageModel` 的类，重写 `score()` 方法。

## 架构说明

### 为什么继承 UserLanguageModel？

`PinyinIME` 的构造函数签名是：
```cpp
PinyinIME(std::unique_ptr<PinyinDictionary> dict,
          std::unique_ptr<UserLanguageModel> model);
```

因此，要替换语言模型，必须创建一个 `UserLanguageModel` 的子类。

### State 管理策略

`State` 类型定义为：
```cpp
constexpr size_t StateSize = 20 + sizeof(void *);
using State = std::array<char, StateSize>;
```

这个大小是为 ngram 模型设计的，不足以存储 BERT 的完整状态。我们的解决方案是：

1. **使用指针存储上下文**：在 `State` 的最后 `sizeof(void*)` 字节中存储指向上下文对象的指针
2. **外部缓存**：使用 `thread_local` 的 `unordered_map` 缓存实际的上下文数据
3. **上下文对象**：`BertStateContext` 存储词历史列表

### 关键方法实现

#### 1. `score()` 方法

这是核心方法，在解码过程中被频繁调用：

```cpp
float score(const State &state, const WordNode &word, State &out) const override {
    // 1. 从 state 中提取上下文指针
    void *context_ptr = extractContextPtr(state);
    
    // 2. 从缓存中获取上下文
    auto context = getContext(context_ptr);
    
    // 3. 使用 BERT 计算分数
    float bert_score = bert_impl_->score(context->words, word.word());
    
    // 4. 可选：与 ngram 混合
    if (use_ngram_fallback_) {
        float ngram_score = UserLanguageModel::score(state, word, ngram_out);
        bert_score = bert_weight_ * bert_score + (1.0f - bert_weight_) * ngram_score;
    }
    
    // 5. 更新输出状态
    updateBertState(state, word, out);
    
    return bert_score;
}
```

#### 2. `updateBertState()` 方法

更新状态，将当前词添加到上下文：

```cpp
void updateBertState(const State &state, const WordNode &word, State &out) const {
    // 1. 获取或创建上下文
    auto context = getOrCreateContext(state);
    
    // 2. 添加当前词到上下文
    context->words.push_back(word.word());
    
    // 3. 限制历史长度（性能优化）
    if (context->words.size() > max_history) {
        context->words.erase(context->words.begin());
    }
    
    // 4. 将上下文指针存储到输出状态
    storeContextPtr(out, context.get());
    
    // 5. 更新缓存
    state_cache_[context.get()] = context;
}
```

## 集成 BERT 推理库

### 选项 1: ONNX Runtime（推荐）

ONNX Runtime 是跨平台的推理引擎，支持多种后端（CPU、GPU、TensorRT 等）。

#### 安装

```bash
# 下载预编译包或从源码编译
# https://github.com/microsoft/onnxruntime/releases
```

#### 集成代码

```cpp
#include <onnxruntime_cxx_api.h>

class BertModelImpl {
    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
public:
    BertModelImpl(const std::string &model_path) 
        : env_(ORT_LOGGING_LEVEL_WARNING, "BertModel"),
          session_(env_, model_path.c_str(), Ort::SessionOptions{nullptr}) {
        
        // 获取输入输出名称
        size_t num_input_nodes = session_.GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            char* input_name = session_.GetInputName(i, allocator_);
            input_names_.push_back(input_name);
        }
        
        size_t num_output_nodes = session_.GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            char* output_name = session_.GetOutputName(i, allocator_);
            output_names_.push_back(output_name);
        }
    }
    
    float score(const std::vector<std::string> &context,
                const std::string &word) const {
        // 1. 准备输入
        std::vector<int64_t> input_ids = tokenize(context, word);
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            allocator_, input_ids.data(), input_ids.size(),
            input_shape.data(), input_shape.size());
        
        // 2. 运行模型
        auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                   input_names_.data(), &input_tensor, 1,
                                   output_names_.data(), 1);
        
        // 3. 提取 logits
        float* logits = outputs[0].GetTensorMutableData<float>();
        int64_t* shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape().data();
        size_t vocab_size = shape[1];
        
        // 4. 找到当前词的 logit
        int word_id = getWordId(word);
        float logit = logits[word_id];
        
        // 5. 转换为 log 概率（简化，实际应该做 softmax）
        return logit / 10.0f;  // 归一化
    }
};
```

### 选项 2: PyTorch C++ API (LibTorch)

如果使用 PyTorch 训练的模型：

```cpp
#include <torch/script.h>

class BertModelImpl {
    torch::jit::script::Module model_;
    
public:
    BertModelImpl(const std::string &model_path) {
        model_ = torch::jit::load(model_path);
        model_.eval();
    }
    
    float score(const std::vector<std::string> &context,
                const std::string &word) const {
        // 准备输入
        auto input_ids = tokenizeToTensor(context, word);
        
        // 运行模型
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_ids);
        auto output = model_.forward(inputs).toTensor();
        
        // 提取分数
        int word_id = getWordId(word);
        float logit = output[0][word_id].item<float>();
        
        return logit / 10.0f;
    }
};
```

### 选项 3: TensorFlow C++ API

```cpp
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>

class BertModelImpl {
    tensorflow::SavedModelBundle bundle_;
    
public:
    BertModelImpl(const std::string &model_path) {
        tensorflow::RunOptions run_options;
        tensorflow::Status status = tensorflow::LoadSavedModel(
            tensorflow::SessionOptions(), run_options, model_path,
            {"serve"}, &bundle_);
    }
    
    float score(const std::vector<std::string> &context,
                const std::string &word) const {
        // 实现类似 ONNX Runtime
    }
};
```

## 性能优化建议

### 1. 批量推理

收集多个候选词，一次性运行 BERT：

```cpp
std::vector<float> batchScore(
    const std::vector<std::string> &context,
    const std::vector<std::string> &candidates) const {
    
    // 准备批量输入
    std::vector<std::vector<int64_t>> batch_inputs;
    for (const auto &word : candidates) {
        batch_inputs.push_back(tokenize(context, word));
    }
    
    // 运行批量推理
    // ...
    
    return batch_scores;
}
```

### 2. 缓存策略

缓存常见上下文的 BERT 输出：

```cpp
class BertModelImpl {
    mutable LRUCache<std::vector<std::string>, std::vector<float>> cache_;
    
    float score(const std::vector<std::string> &context,
                const std::string &word) const {
        // 检查缓存
        auto cached = cache_.get(context);
        if (cached) {
            return (*cached)[getWordId(word)];
        }
        
        // 计算并缓存
        auto scores = computeScores(context);
        cache_.put(context, scores);
        return scores[getWordId(word)];
    }
};
```

### 3. 异步推理

在后台线程运行 BERT，不阻塞输入：

```cpp
class AsyncBertModel {
    std::thread worker_thread_;
    std::queue<ScoreRequest> request_queue_;
    std::mutex queue_mutex_;
    
public:
    void scoreAsync(const State &state, const WordNode &word,
                   std::function<void(float)> callback) {
        // 添加到队列
        // 后台线程处理
    }
};
```

### 4. 模型量化

使用 INT8 量化模型可以显著提升速度：

```python
# 在 Python 中量化模型
import onnx
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic("bert_model.onnx", "bert_model_int8.onnx")
```

## 词汇表处理

### 词汇表对齐

BERT 词汇表可能与输入法词典不匹配，需要处理：

1. **映射表**：创建 BERT 词汇表到输入法词典的映射
2. **未知词处理**：对于不在 BERT 词汇表中的词，使用 UNK token
3. **子词分割**：如果使用 WordPiece/BPE，需要正确分割

```cpp
class VocabularyMapper {
    std::unordered_map<std::string, int> bert_vocab_;
    std::unordered_map<std::string, std::vector<int>> word_to_tokens_;
    
public:
    std::vector<int> tokenize(const std::string &word) const {
        // 如果词在词汇表中，直接返回
        if (bert_vocab_.count(word)) {
            return {bert_vocab_.at(word)};
        }
        
        // 否则进行子词分割
        return wordPieceSplit(word);
    }
};
```

## 测试

### 单元测试

```cpp
#include <gtest/gtest.h>

TEST(BertLanguageModel, BasicScore) {
    auto model = std::make_unique<BertLanguageModel>(
        "model.onnx", "vocab.txt", "dummy.lm");
    
    State state = model->beginState();
    State out;
    WordNode word("你好", 0);
    
    float score = model->score(state, word, out);
    EXPECT_GT(score, -100.0f);
    EXPECT_LT(score, 0.0f);  // Log probability should be negative
}
```

### 集成测试

```cpp
TEST(BertLanguageModel, PinyinIMEIntegration) {
    auto dict = std::make_unique<PinyinDictionary>();
    dict->load("test.dict");
    
    auto bert_model = std::make_unique<BertLanguageModel>(
        "model.onnx", "vocab.txt", "dummy.lm");
    
    PinyinIME ime(std::move(dict), std::move(bert_model));
    PinyinContext context(&ime);
    
    context.type("nihao");
    const auto &candidates = context.candidates();
    
    EXPECT_GT(candidates.size(), 0);
    EXPECT_EQ(candidates[0].toString(), "你好");
}
```

## 常见问题

### Q: 为什么需要 dummy ngram 文件？

A: `UserLanguageModel` 的构造函数需要一个 ngram 文件。可以传入空文件或最小化的文件，因为我们不会真正使用它。

### Q: State 空间不够怎么办？

A: 使用指针指向外部缓存。这是当前实现采用的方法。

### Q: 性能如何？

A: BERT 比 ngram 慢很多。建议：
- 使用量化模型
- 限制 beam search 宽度
- 使用 GPU 加速
- 实现缓存

### Q: 可以完全替换 ngram 吗？

A: 可以，设置 `setUseNgramFallback(false)` 即可。但建议保留混合模式以获得更好的鲁棒性。

## 下一步

1. 选择并集成 BERT 推理库（推荐 ONNX Runtime）
2. 实现实际的 `BertModelImpl::score()` 方法
3. 处理词汇表对齐和子词分割
4. 实现性能优化（缓存、批量推理等）
5. 进行测试和调优

