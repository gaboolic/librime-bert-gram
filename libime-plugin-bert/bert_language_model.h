/*
 * BERT Language Model Plugin for libime
 * 
 * This plugin allows replacing ngram language model with BERT model
 * without modifying libime source code.
 */

#ifndef BERT_LANGUAGE_MODEL_H_
#define BERT_LANGUAGE_MODEL_H_

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <libime/core/userlanguagemodel.h>
#include <libime/core/languagemodel.h>
#include <libime/core/constants.h>

namespace libime {

// Forward declaration for BERT model implementation
class BertModelImpl;

/**
 * BERT-based Language Model that extends UserLanguageModel
 * 
 * This class wraps a BERT model and provides the LanguageModelBase interface.
 * It can be used as a drop-in replacement for UserLanguageModel in PinyinIME.
 */
class BertLanguageModel : public UserLanguageModel {
public:
    /**
     * Constructor
     * @param bert_model_path Path to the BERT model file
     * @param vocab_path Path to the vocabulary file
     * @param sysfile Path to a dummy ngram file (required by UserLanguageModel, can be empty)
     */
    explicit BertLanguageModel(const std::string &bert_model_path,
                               const std::string &vocab_path,
                               const char *sysfile = nullptr);

    virtual ~BertLanguageModel();

    /**
     * Override score method to use BERT instead of ngram
     */
    float score(const State &state, const WordNode &word,
                State &out) const override;

    /**
     * Override beginState to return BERT-compatible state
     */
    const State &beginState() const override;

    /**
     * Override nullState to return BERT-compatible state
     */
    const State &nullState() const override;

    /**
     * Get word index from BERT vocabulary
     */
    WordIndex index(std::string_view view) const override;

    /**
     * Check if word is unknown in BERT vocabulary
     */
    bool isUnknown(WordIndex idx, std::string_view view) const override;

    /**
     * Set BERT model parameters
     */
    void setBertWeight(float weight);
    float bertWeight() const;

    /**
     * Enable/disable fallback to ngram model
     */
    void setUseNgramFallback(bool use);
    bool useNgramFallback() const;

private:
    std::unique_ptr<BertModelImpl> bert_impl_;
    float bert_weight_ = 1.0f;
    bool use_ngram_fallback_ = false;
    
    // BERT-specific state management
    mutable State bert_begin_state_;
    mutable State bert_null_state_;
    
    // Helper methods
    float computeBertScore(const State &state, const WordNode &word,
                          State &out) const;
    void updateBertState(const State &state, const WordNode &word,
                        State &out) const;
};

} // namespace libime

#endif // BERT_LANGUAGE_MODEL_H_

