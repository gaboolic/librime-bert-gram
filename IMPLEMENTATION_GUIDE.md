# BERT Grammar 实现指南

## 快速开始

### 方案 1: 使用 ONNX Runtime（推荐）

ONNX Runtime 是一个高性能的跨平台推理引擎，支持多种硬件加速。

#### 1. 安装 ONNX Runtime

```bash
# 下载预编译库
# 从 https://github.com/microsoft/onnxruntime/releases 下载
# 或使用包管理器安装
```

#### 2. 修改 CMakeLists.txt

```cmake
find_package(onnxruntime REQUIRED)
target_link_libraries(rime-bert-grammar-objs
  ${ONNXRUNTIME_LIBRARIES})
target_include_directories(rime-bert-grammar-objs PRIVATE
  ${ONNXRUNTIME_INCLUDE_DIRS})
```

#### 3. 实现 ONNX 推理

在 `bert_grammar.cc` 中：

```cpp
#include <onnxruntime_cxx_api.h>

class BertGrammar::Impl {
 public:
  Ort::Env env_;
  Ort::Session session_{nullptr};
  Ort::MemoryInfo memory_info_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  
  bool LoadModel(const string& model_path) {
    Ort::SessionOptions session_options;
    session_ = Ort::Session(env_, model_path.c_str(), session_options);
    
    // Get input/output names
    size_t num_input_nodes = session_.GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
      char* input_name = session_.GetInputName(i, allocator_);
      input_names_.push_back(input_name);
    }
    // Similar for output names...
    return true;
  }
  
  double ComputeProbability(const string& context, const string& word) {
    // 1. Tokenize
    auto context_tokens = Tokenize(context);
    auto word_tokens = Tokenize(word);
    
    // 2. Build input: [CLS] context [SEP] word [SEP]
    std::vector<int64_t> input_ids;
    input_ids.push_back(101);  // [CLS]
    for (int token : context_tokens) {
      input_ids.push_back(token);
    }
    input_ids.push_back(102);  // [SEP]
    for (int token : word_tokens) {
      input_ids.push_back(token);
    }
    input_ids.push_back(102);  // [SEP]
    
    // 3. Create input tensor
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info_, input_ids.data(), input_ids.size(),
        input_shape.data(), input_shape.size());
    
    // 4. Run inference
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), &input_tensor, 1,
        output_names_.data(), 1);
    
    // 5. Extract probability
    float* output = output_tensors[0].GetTensorMutableData<float>();
    // Process output to get probability...
    
    return probability;
  }
};
```

### 方案 2: 使用 PyTorch C++ API (LibTorch)

#### 1. 安装 LibTorch

```bash
# 从 https://pytorch.org/get-started/locally/ 下载
```

#### 2. 修改 CMakeLists.txt

```cmake
find_package(Torch REQUIRED)
target_link_libraries(rime-bert-grammar-objs
  ${TORCH_LIBRARIES})
target_include_directories(rime-bert-grammar-objs PRIVATE
  ${TORCH_INCLUDE_DIRS})
```

#### 3. 实现 LibTorch 推理

```cpp
#include <torch/script.h>

class BertGrammar::Impl {
  torch::jit::script::Module model_;
  
  bool LoadModel(const string& model_path) {
    try {
      model_ = torch::jit::load(model_path);
      model_.eval();
      return true;
    } catch (const c10::Error& e) {
      LOG(ERROR) << "Error loading model: " << e.what();
      return false;
    }
  }
  
  double ComputeProbability(const string& context, const string& word) {
    // Tokenize and create input
    auto input_ids = TokenizeToTensor(context, word);
    
    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_ids);
    auto output = model_.forward(inputs).toTensor();
    
    // Extract probability
    auto prob = output[0].item<double>();
    return prob;
  }
};
```

### 方案 3: 通过 HTTP API 调用 BERT 服务

如果不想在本地运行 BERT 模型，可以通过 HTTP API 调用远程服务：

```cpp
#include <curl/curl.h>

double BertGrammar::ComputeProbability(const string& context,
                                      const string& word) {
  // Build JSON request
  json request;
  request["context"] = context;
  request["word"] = word;
  
  // Send HTTP POST request
  CURL* curl = curl_easy_init();
  // ... configure curl ...
  curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8000/predict");
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request.dump().c_str());
  
  string response;
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
  curl_easy_perform(curl);
  
  // Parse response
  json result = json::parse(response);
  return result["probability"].get<double>();
}
```

## BERT Tokenizer 实现

你需要实现一个 BERT tokenizer，或者使用现有的库：

### 选项 1: 使用 HuggingFace tokenizers (C++)

```cpp
#include <tokenizers_c.h>

class BertTokenizer {
  TokenizerPtr tokenizer_;
  
 public:
  bool Load(const string& vocab_path) {
    tokenizer_ = from_file(vocab_path.c_str());
    return tokenizer_ != nullptr;
  }
  
  std::vector<int> Encode(const string& text) {
    // Use tokenizer to encode
  }
};
```

### 选项 2: 自己实现简单的 WordPiece tokenizer

```cpp
std::vector<int> BertGrammar::Tokenize(const string& text) {
  std::vector<int> tokens;
  // Simple implementation:
  // 1. Split by whitespace
  // 2. For each word, apply WordPiece tokenization
  // 3. Convert to token IDs using vocab_map_
  return tokens;
}
```

## 模型转换

如果你有 PyTorch 或 TensorFlow 的 BERT 模型，需要转换为推理格式：

### PyTorch -> ONNX

```python
import torch
from transformers import BertForMaskedLM

model = BertForMaskedLM.from_pretrained('bert-base-chinese')
model.eval()

dummy_input = torch.randint(0, 21128, (1, 128))
torch.onnx.export(
    model,
    dummy_input,
    "bert_model.onnx",
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                  'logits': {0: 'batch_size', 1: 'sequence'}}
)
```

### TensorFlow -> ONNX

使用 `tf2onnx`:

```bash
python -m tf2onnx.convert --saved-model saved_model_dir --output bert_model.onnx
```

## 性能优化建议

1. **模型量化**: 使用 INT8 量化减少模型大小和加速
2. **批处理**: 如果可能，批量处理多个查询
3. **缓存**: 缓存常见 context-word 对的概率
4. **异步推理**: 使用异步推理避免阻塞主线程
5. **使用 GPU**: 如果可用，使用 GPU 加速推理

## 测试

创建测试用例确保实现正确：

```cpp
TEST(BertGrammarTest, BasicQuery) {
  Config config;
  config.SetString("bert_grammar/model_path", "test_model.onnx");
  config.SetString("bert_grammar/vocab_path", "test_vocab.txt");
  
  BertGrammar grammar(&config);
  double prob = grammar.Query("今天", "天气");
  EXPECT_GT(prob, -20.0);  // Should be a reasonable log probability
}
```

