//
// Copyright RIME Developers
// Distributed under the BSD License
//
// BERT Grammar implementation using ONNX Runtime
// This is an example implementation - you can adapt it to your BERT library
//

#ifndef RIME_BERT_GRAMMAR_ONNX_H_
#define RIME_BERT_GRAMMAR_ONNX_H_

#include "bert_grammar.h"
#include <memory>
#include <string>
#include <vector>

// Forward declarations for ONNX Runtime
// Uncomment and include when you have ONNX Runtime installed
// #include <onnxruntime_cxx_api.h>

namespace rime {

class BertGrammarOnnx : public BertGrammar {
 public:
  explicit BertGrammarOnnx(Config* config);
  virtual ~BertGrammarOnnx();

 protected:
  bool LoadModel() override;
  double ComputeProbability(const string& context,
                           const string& word) override;

 private:
  // ONNX Runtime session
  // Uncomment when using ONNX Runtime:
  // std::unique_ptr<Ort::Session> session_;
  // Ort::Env env_;
  // Ort::MemoryInfo memory_info_;
  
  // Tokenizer
  std::vector<int> Tokenize(const string& text);
  string Detokenize(const std::vector<int>& tokens);
  
  // Vocabulary
  std::vector<string> vocab_;
  std::map<string, int> vocab_map_;
  
  bool LoadVocabulary(const string& vocab_path);
};

}  // namespace rime

#endif  // RIME_BERT_GRAMMAR_ONNX_H_


