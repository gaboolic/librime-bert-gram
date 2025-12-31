/*
 * Example usage of BERT Language Model Plugin
 * 
 * This shows how to use BertLanguageModel as a drop-in replacement
 * for UserLanguageModel in PinyinIME.
 */

#include "bert_language_model.h"
#include <libime/pinyin/pinyinime.h>
#include <libime/pinyin/pinyindictionary.h>
#include <memory>
#include <iostream>

int main() {
    // 1. Create BERT language model
    // Note: You need to provide paths to your BERT model and vocabulary
    auto bert_model = std::make_unique<libime::BertLanguageModel>(
        "/path/to/bert/model.onnx",      // BERT model path
        "/path/to/vocab.txt",            // Vocabulary file
        "/path/to/dummy.lm"              // Dummy ngram file (can be empty)
    );
    
    // Configure BERT model
    bert_model->setBertWeight(1.0f);           // Use 100% BERT
    bert_model->setUseNgramFallback(false);    // Don't use ngram fallback
    
    // 2. Create dictionary (same as before)
    auto dict = std::make_unique<libime::PinyinDictionary>();
    dict->load("/path/to/dict.dict");
    
    // 3. Create PinyinIME with BERT model
    // This works because BertLanguageModel inherits from UserLanguageModel
    libime::PinyinIME ime(std::move(dict), std::move(bert_model));
    
    // 4. Use PinyinIME as normal
    // The BERT model will be used automatically in the scoring process
    auto context = std::make_unique<libime::PinyinContext>(&ime);
    
    // Type some pinyin
    context->type("nihao");
    
    // Get candidates (scored by BERT)
    const auto &candidates = context->candidates();
    for (size_t i = 0; i < candidates.size(); ++i) {
        std::cout << "Candidate " << i << ": " 
                  << candidates[i].toString() 
                  << " (score: " << candidates[i].score() << ")" 
                  << std::endl;
    }
    
    return 0;
}

