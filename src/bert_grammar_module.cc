//
// Copyright RIME Developers
// Distributed under the BSD License
//
// BERT Grammar module registration
//

#include <rime/component.h>
#include <rime/registry.h>
#include <rime_api.h>
#include "bert_grammar.h"

using namespace rime;

static void rime_bert_grammar_initialize() {
  LOG(INFO) << "registering BERT grammar component.";
  Registry& r = Registry::instance();
  
  // Register as "grammar" to replace octagram
  // Note: This will override any existing "grammar" registration
  r.Register("grammar", new Component<BertGrammar>);
}

static void rime_bert_grammar_finalize() {
  // Cleanup if needed
}

RIME_REGISTER_MODULE(bert_grammar)


