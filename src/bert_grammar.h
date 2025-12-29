//
// Copyright RIME Developers
// Distributed under the BSD License
//
// BERT-based Grammar component for Rime using ONNX Runtime
//

#ifndef RIME_BERT_GRAMMAR_H_
#define RIME_BERT_GRAMMAR_H_

#include <rime/common.h>
#include <rime/config.h>
#include <rime/gear/grammar.h>
#include <memory>
#include <vector>
#include <map>
#include <string>

// Forward declarations for ONNX Runtime
// Include the actual header in .cc file to avoid exposing ONNX dependencies
namespace Ort {
  class Env;
  class Session;
  class MemoryInfo;
  class AllocatorWithDefaultOptions;
}

namespace rime {

class BertGrammar : public Grammar {
 public:
  explicit BertGrammar(Config* config);
  virtual ~BertGrammar();

  // Implement Grammar interface
  double Query(const string& context,
               const string& word,
               bool is_rear) override;

 private:
  // Initialize BERT model
  bool LoadModel();
  
  // Call BERT model to compute probability
  double ComputeProbability(const string& context, const string& word);
  
  // Tokenization
  std::vector<int64_t> Tokenize(const string& text);
  bool LoadVocabulary(const string& vocab_path);
  int GetTokenId(const string& token);
  
  // ONNX Runtime inference
  bool RunInference(const std::vector<int64_t>& input_ids,
                    std::vector<float>& output);
  
  // BERT model path and configuration
  string model_path_;
  string vocab_path_;
  bool model_loaded_;
  
  // ONNX Runtime objects (using pimpl to hide implementation)
  class Impl;
  std::unique_ptr<Impl> impl_;
  
  // Vocabulary
  std::vector<string> vocab_;
  std::map<string, int> vocab_map_;
  static const int kUnkTokenId = 100;  // [UNK] token ID
  static const int kClsTokenId = 101;  // [CLS] token ID
  static const int kSepTokenId = 102;  // [SEP] token ID
  static const int kPadTokenId = 0;    // [PAD] token ID
  static const int kMaxSequenceLength = 512;
};

}  // namespace rime

#endif  // RIME_BERT_GRAMMAR_H_

