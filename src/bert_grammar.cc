//
// Copyright RIME Developers
// Distributed under the BSD License
//
// BERT-based Grammar component implementation using ONNX Runtime
//

#include "bert_grammar.h"
#include <rime/config.h>
#include <rime/deployer.h>
#include <rime/service.h>
#include <glog/logging.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

// Include ONNX Runtime headers
#ifdef RIME_USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace rime {

// Pimpl implementation to hide ONNX Runtime details
class BertGrammar::Impl {
 public:
#ifdef RIME_USE_ONNXRUNTIME
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_info_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  std::vector<Ort::Value> input_tensors_;
  
  Impl() : env_(ORT_LOGGING_LEVEL_WARNING, "BertGrammar"),
           memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  }
  
  ~Impl() {
    session_.reset();
    // Clean up string names
    for (auto* name : input_names_) {
      allocator_.Free(const_cast<void*>(reinterpret_cast<const void*>(name)));
    }
    for (auto* name : output_names_) {
      allocator_.Free(const_cast<void*>(reinterpret_cast<const void*>(name)));
    }
  }
  
  bool LoadSession(const string& model_path) {
    try {
      Ort::SessionOptions session_options;
      // Enable optimizations
      session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_ALL);
      
      // Try to use CUDA if available (optional)
      // Uncomment if you have CUDA support:
      // OrtCUDAProviderOptions cuda_options{};
      // session_options.AppendExecutionProvider_CUDA(cuda_options);
      
      session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
      
      // Get input/output names
      size_t num_input_nodes = session_->GetInputCount();
      size_t num_output_nodes = session_->GetOutputCount();
      
      for (size_t i = 0; i < num_input_nodes; i++) {
        char* input_name = session_->GetInputName(i, allocator_);
        input_names_.push_back(input_name);
      }
      
      for (size_t i = 0; i < num_output_nodes; i++) {
        char* output_name = session_->GetOutputName(i, allocator_);
        output_names_.push_back(output_name);
      }
      
      LOG(INFO) << "ONNX Runtime session loaded successfully";
      LOG(INFO) << "Input nodes: " << num_input_nodes;
      LOG(INFO) << "Output nodes: " << num_output_nodes;
      
      return true;
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to load ONNX model: " << e.what();
      return false;
    }
  }
  
  bool RunInference(const std::vector<int64_t>& input_ids,
                    std::vector<float>& output) {
    if (!session_) {
      return false;
    }
    
    try {
      // Create input tensor
      std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};
      Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
          memory_info_,
          const_cast<int64_t*>(input_ids.data()),
          input_ids.size(),
          input_shape.data(),
          input_shape.size());
      
      // Create attention mask (all ones for now)
      std::vector<int64_t> attention_mask(input_ids.size(), 1);
      Ort::Value attention_tensor = Ort::Value::CreateTensor<int64_t>(
          memory_info_,
          attention_mask.data(),
          attention_mask.size(),
          input_shape.data(),
          input_shape.size());
      
      // Prepare inputs
      std::vector<Ort::Value> ort_inputs;
      ort_inputs.push_back(std::move(input_tensor));
      // Add attention mask if model expects it
      if (input_names_.size() > 1) {
        ort_inputs.push_back(std::move(attention_tensor));
      }
      
      // Run inference
      auto output_tensors = session_->Run(
          Ort::RunOptions{nullptr},
          input_names_.data(),
          ort_inputs.data(),
          ort_inputs.size(),
          output_names_.data(),
          output_names_.size());
      
      // Extract output
      if (output_tensors.empty()) {
        return false;
      }
      
      float* output_data = output_tensors[0].GetTensorMutableData<float>();
      auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
      size_t output_size = 1;
      for (auto dim : output_shape) {
        output_size *= dim;
      }
      
      output.assign(output_data, output_data + output_size);
      return true;
    } catch (const std::exception& e) {
      LOG(ERROR) << "Inference failed: " << e.what();
      return false;
    }
  }
#else
  // Stub implementation when ONNX Runtime is not available
  Impl() {}
  ~Impl() {}
  bool LoadSession(const string&) { return false; }
  bool RunInference(const std::vector<int64_t>&, std::vector<float>&) { return false; }
#endif
};

BertGrammar::BertGrammar(Config* config)
    : model_loaded_(false), impl_(std::make_unique<Impl>()) {
  if (!config) {
    LOG(ERROR) << "BertGrammar: config is null";
    return;
  }
  
  // Load configuration
  config->GetString("bert_grammar/model_path", &model_path_);
  config->GetString("bert_grammar/vocab_path", &vocab_path_);
  
  // If paths are relative, resolve them relative to shared data dir
  if (!model_path_.empty() && model_path_[0] != '/' && model_path_[1] != ':') {
    Deployer& deployer = Service::instance().deployer();
    model_path_ = (deployer.shared_data_dir / model_path_).string();
  }
  if (!vocab_path_.empty() && vocab_path_[0] != '/' && vocab_path_[1] != ':') {
    Deployer& deployer = Service::instance().deployer();
    vocab_path_ = (deployer.shared_data_dir / vocab_path_).string();
  }
  
  // Load vocabulary first
  if (!vocab_path_.empty()) {
    if (!LoadVocabulary(vocab_path_)) {
      LOG(WARNING) << "BertGrammar: failed to load vocabulary from " << vocab_path_;
    }
  }
  
  // Load BERT model
  if (!model_path_.empty()) {
    model_loaded_ = LoadModel();
    if (!model_loaded_) {
      LOG(ERROR) << "BertGrammar: failed to load BERT model from " << model_path_;
    } else {
      LOG(INFO) << "BertGrammar: model loaded successfully";
    }
  } else {
    LOG(WARNING) << "BertGrammar: model_path not configured";
  }
}

BertGrammar::~BertGrammar() {
  // Impl will be automatically destroyed
}

bool BertGrammar::LoadModel() {
#ifdef RIME_USE_ONNXRUNTIME
  if (model_path_.empty()) {
    return false;
  }
  
  return impl_->LoadSession(model_path_);
#else
  LOG(ERROR) << "ONNX Runtime support not compiled in. "
             << "Define RIME_USE_ONNXRUNTIME and link against ONNX Runtime.";
  return false;
#endif
}

bool BertGrammar::LoadVocabulary(const string& vocab_path) {
  std::ifstream file(vocab_path);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open vocabulary file: " << vocab_path;
    return false;
  }
  
  vocab_.clear();
  vocab_map_.clear();
  
  string line;
  int id = 0;
  while (std::getline(file, line)) {
    // Remove BOM if present
    if (!line.empty() && line[0] == '\xEF' && line.size() >= 3 &&
        line[1] == '\xBB' && line[2] == '\xBF') {
      line = line.substr(3);
    }
    // Trim whitespace
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    
    if (!line.empty()) {
      vocab_.push_back(line);
      vocab_map_[line] = id;
      id++;
    }
  }
  
  LOG(INFO) << "Loaded vocabulary with " << vocab_.size() << " tokens";
  return !vocab_.empty();
}

int BertGrammar::GetTokenId(const string& token) {
  auto it = vocab_map_.find(token);
  if (it != vocab_map_.end()) {
    return it->second;
  }
  // Try to find subword tokens (WordPiece style)
  // For now, return UNK if not found
  return kUnkTokenId;
}

std::vector<int64_t> BertGrammar::Tokenize(const string& text) {
  std::vector<int64_t> token_ids;
  
  if (vocab_.empty()) {
    // Fallback: simple character-based tokenization
    // This is not ideal but works as a fallback
    for (size_t i = 0; i < text.length();) {
      // Handle UTF-8 characters
      unsigned char c = text[i];
      if (c < 0x80) {
        // ASCII
        string token(1, c);
        token_ids.push_back(GetTokenId(token));
        i++;
      } else if ((c & 0xE0) == 0xC0) {
        // 2-byte UTF-8
        if (i + 1 < text.length()) {
          string token = text.substr(i, 2);
          token_ids.push_back(GetTokenId(token));
          i += 2;
        } else {
          i++;
        }
      } else if ((c & 0xF0) == 0xE0) {
        // 3-byte UTF-8
        if (i + 2 < text.length()) {
          string token = text.substr(i, 3);
          token_ids.push_back(GetTokenId(token));
          i += 3;
        } else {
          i++;
        }
      } else if ((c & 0xF8) == 0xF0) {
        // 4-byte UTF-8
        if (i + 3 < text.length()) {
          string token = text.substr(i, 4);
          token_ids.push_back(GetTokenId(token));
          i += 4;
        } else {
          i++;
        }
      } else {
        i++;
      }
    }
  } else {
    // Simple WordPiece-like tokenization
    // Split by whitespace first
    std::istringstream iss(text);
    string word;
    while (iss >> word) {
      // Try to match the word in vocabulary
      int token_id = GetTokenId(word);
      if (token_id != kUnkTokenId) {
        token_ids.push_back(token_id);
      } else {
        // WordPiece: try to split into subwords
        // Simple implementation: try common prefixes
        bool found = false;
        for (size_t len = word.length(); len > 0 && !found; len--) {
          string subword = word.substr(0, len);
          if (vocab_map_.find(subword) != vocab_map_.end()) {
            token_ids.push_back(GetTokenId(subword));
            // Recursively tokenize the rest
            if (len < word.length()) {
              string remaining = "##" + word.substr(len);
              auto remaining_tokens = Tokenize(remaining);
              token_ids.insert(token_ids.end(), remaining_tokens.begin(), remaining_tokens.end());
            }
            found = true;
          }
        }
        if (!found) {
          token_ids.push_back(kUnkTokenId);
        }
      }
    }
  }
  
  return token_ids;
}

double BertGrammar::Query(const string& context,
                          const string& word,
                          bool is_rear) {
  // If model is not loaded, return a default penalty
  if (!model_loaded_ || model_path_.empty()) {
    return -18.420680743952367;  // log(1e-8)
  }
  
  // Compute probability using BERT model
  double prob = ComputeProbability(context, word);
  
  // Return log probability
  // Ensure probability is positive to avoid log(0)
  if (prob <= 0) {
    prob = 1e-8;
  }
  return log(prob);
}

double BertGrammar::ComputeProbability(const string& context,
                                       const string& word) {
#ifdef RIME_USE_ONNXRUNTIME
  if (!model_loaded_ || !impl_->session_) {
    return 1e-8;
  }
  
  // 1. Tokenize context and word
  auto context_tokens = Tokenize(context);
  auto word_tokens = Tokenize(word);
  
  // 2. Build input sequence: [CLS] context [SEP] word [SEP]
  std::vector<int64_t> input_ids;
  input_ids.push_back(kClsTokenId);  // [CLS]
  
  // Add context tokens
  for (int64_t token : context_tokens) {
    input_ids.push_back(token);
  }
  input_ids.push_back(kSepTokenId);  // [SEP]
  
  // Add word tokens
  for (int64_t token : word_tokens) {
    input_ids.push_back(token);
  }
  input_ids.push_back(kSepTokenId);  // [SEP]
  
  // Truncate if too long
  if (input_ids.size() > kMaxSequenceLength) {
    // Keep [CLS], context (truncated), [SEP], word, [SEP]
    size_t context_len = context_tokens.size();
    size_t word_len = word_tokens.size();
    size_t max_context_len = kMaxSequenceLength - word_len - 3;  // -3 for [CLS] and 2 [SEP]
    
    if (max_context_len > 0 && context_len > max_context_len) {
      input_ids.clear();
      input_ids.push_back(kClsTokenId);
      for (size_t i = context_len - max_context_len; i < context_len; i++) {
        input_ids.push_back(context_tokens[i]);
      }
      input_ids.push_back(kSepTokenId);
      for (int64_t token : word_tokens) {
        input_ids.push_back(token);
      }
      input_ids.push_back(kSepTokenId);
    }
  }
  
  // 3. Run inference
  std::vector<float> output;
  if (!impl_->RunInference(input_ids, output)) {
    LOG(WARNING) << "BERT inference failed";
    return 1e-8;
  }
  
  // 4. Extract probability from output
  // The output format depends on your BERT model architecture
  // For a typical BERT model used for language modeling:
  // - Output shape might be [batch, sequence_length, vocab_size] or [batch, vocab_size]
  // - We need to extract the probability of the word given the context
  
  if (output.empty()) {
    return 1e-8;
  }
  
  // Simple approach: use the CLS token's embedding or the last token's logits
  // For a proper implementation, you might want to:
  // 1. Use the logits at the position of the word
  // 2. Apply softmax to get probabilities
  // 3. Extract the probability of the specific word token
  
  // For now, use a heuristic: take the mean of output values as a score
  // This is a simplified approach - you should adapt based on your model's output format
  double score = 0.0;
  for (float val : output) {
    score += val;
  }
  score /= output.size();
  
  // Convert score to probability (using sigmoid or softmax)
  // This is a simplified conversion - adjust based on your model
  double prob = 1.0 / (1.0 + exp(-score));
  
  // Normalize to a reasonable range
  if (prob < 1e-8) {
    prob = 1e-8;
  }
  
  return prob;
#else
  LOG(WARNING) << "ONNX Runtime not available, returning default probability";
  return 1e-8;
#endif
}

}  // namespace rime
