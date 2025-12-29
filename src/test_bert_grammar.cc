//
// Copyright RIME Developers
// Distributed under the BSD License
//
// 测试程序：验证 BERT Grammar 组件是否正常工作
//

// Define glog macros before including headers
// glog requires GLOG_EXPORT and GLOG_NO_EXPORT to be defined
#define GLOG_EXPORT
#define GLOG_NO_EXPORT
// Define GLOG_DEPRECATED if not defined
#ifndef GLOG_DEPRECATED
#define GLOG_DEPRECATED
#endif
// Don't define GLOG_USE_GLOG_EXPORT to avoid including non-existent export.h

// Include rime headers first
#include "bert_grammar.h"
#include <rime/config.h>
#include <rime/deployer.h>
#include <rime/service.h>
// Include glog after rime headers
#include <glog/logging.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <fstream>
#include <cstdio>

using namespace rime;

// 测试用例结构
struct TestCase {
  std::string context;
  std::string word;
  std::string description;
};

// 打印分隔线
void PrintSeparator(const std::string& title = "") {
  std::cout << "\n" << std::string(60, '=') << "\n";
  if (!title.empty()) {
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n";
  }
}

// 测试模型加载
bool TestModelLoading(BertGrammar* grammar) {
  PrintSeparator("测试 1: 模型加载状态");
  
  // 检查模型是否已加载
  // 注意：BertGrammar 没有公开的 is_loaded() 方法
  // 我们可以通过调用 Query 来间接测试
  std::cout << "正在检查模型加载状态...\n";
  
  // 尝试一个简单的查询
  double result = grammar->Query("测试", "模型", false);
  
  // 如果返回的是默认值 -18.42 (log(1e-8))，说明模型可能未加载
  const double default_penalty = -18.420680743952367;
  const double tolerance = 0.001;
  
  if (std::abs(result - default_penalty) < tolerance) {
    std::cout << "⚠ 警告: 返回值为默认惩罚值，模型可能未正确加载\n";
    std::cout << "   返回值: " << std::fixed << std::setprecision(6) << result << "\n";
    std::cout << "   默认值: " << default_penalty << "\n";
    return false;
  } else {
    std::cout << "✓ 模型似乎已加载（返回值不是默认值）\n";
    std::cout << "   返回值: " << std::fixed << std::setprecision(6) << result << "\n";
    return true;
  }
}

// 测试词汇表加载
bool TestVocabulary(BertGrammar* grammar) {
  PrintSeparator("测试 2: 词汇表功能");
  
  // 测试一些常见的中文词汇
  std::vector<std::string> test_words = {
    "你好", "世界", "测试", "模型", "中文"
  };
  
  std::cout << "测试词汇识别...\n";
  bool all_found = true;
  
  for (const auto& word : test_words) {
    double score = grammar->Query("", word, false);
    std::cout << "  \"" << word << "\": " 
              << std::fixed << std::setprecision(6) << score;
    
    // 检查是否返回有效值（不是 NaN 或 Inf）
    if (std::isnan(score) || std::isinf(score)) {
      std::cout << " [错误: 无效值]";
      all_found = false;
    } else {
      std::cout << " [正常]";
    }
    std::cout << "\n";
  }
  
  return all_found;
}

// 测试上下文相关的查询
bool TestContextualQueries(BertGrammar* grammar) {
  PrintSeparator("测试 3: 上下文相关查询");
  
  std::vector<TestCase> test_cases = {
    {"今天", "天气", "简单上下文"},
    {"我喜欢", "吃", "动词上下文"},
    {"各个国家有各个国家的", "国歌", "长上下文"},
    {"", "你好", "无上下文"},
    {"北京是", "中国", "地理上下文"},
  };
  
  std::cout << "测试不同上下文下的词条评分...\n\n";
  
  bool all_passed = true;
  for (const auto& test : test_cases) {
    double score = grammar->Query(test.context, test.word, false);
    
    std::cout << "上下文: \"" << (test.context.empty() ? "(空)" : test.context) << "\"\n";
    std::cout << "词条: \"" << test.word << "\"\n";
    std::cout << "描述: " << test.description << "\n";
    std::cout << "评分: " << std::fixed << std::setprecision(6) << score;
    
    // 检查返回值是否有效
    if (std::isnan(score) || std::isinf(score)) {
      std::cout << " [错误: 无效值]";
      all_passed = false;
    } else if (score > 0) {
      std::cout << " [警告: 正值，通常应该是负值（对数概率）]";
    } else {
      std::cout << " [正常]";
    }
    std::cout << "\n\n";
  }
  
  return all_passed;
}

// 测试对比：相同词条在不同上下文下的评分应该不同
bool TestContextSensitivity(BertGrammar* grammar) {
  PrintSeparator("测试 4: 上下文敏感性");
  
  std::cout << "测试相同词条在不同上下文下的评分差异...\n\n";
  
  const std::string word = "苹果";
  std::vector<std::string> contexts = {
    "我喜欢吃",
    "我在看",
    "我买了一个",
    ""
  };
  
  std::vector<double> scores;
  for (const auto& context : contexts) {
    double score = grammar->Query(context, word, false);
    scores.push_back(score);
    
    std::cout << "上下文: \"" << (context.empty() ? "(空)" : context) << "\"\n";
    std::cout << "词条: \"" << word << "\"\n";
    std::cout << "评分: " << std::fixed << std::setprecision(6) << score << "\n\n";
  }
  
  // 检查不同上下文是否产生不同的评分
  bool has_variation = false;
  for (size_t i = 0; i < scores.size(); i++) {
    for (size_t j = i + 1; j < scores.size(); j++) {
      if (std::abs(scores[i] - scores[j]) > 0.01) {
        has_variation = true;
        break;
      }
    }
    if (has_variation) break;
  }
  
  if (has_variation) {
    std::cout << "✓ 不同上下文产生了不同的评分（模型正常工作）\n";
  } else {
    std::cout << "⚠ 警告: 不同上下文产生了相似的评分\n";
    std::cout << "   这可能表示模型未正确使用上下文信息\n";
  }
  
  return has_variation;
}

// 性能测试
void TestPerformance(BertGrammar* grammar) {
  PrintSeparator("测试 5: 性能测试");
  
  std::cout << "运行 10 次查询以测试性能...\n";
  
  const std::string context = "今天天气很好";
  const std::string word = "我们";
  
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < 10; i++) {
    grammar->Query(context, word, false);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  double avg_time_ms = duration.count() / 10000.0;  // 转换为毫秒并取平均
  
  std::cout << "10 次查询总耗时: " << duration.count() / 1000.0 << " ms\n";
  std::cout << "平均每次查询: " << std::fixed << std::setprecision(2) 
            << avg_time_ms << " ms\n";
  std::cout << "吞吐量: " << std::fixed << std::setprecision(1) 
            << (1000.0 / avg_time_ms) << " 查询/秒\n";
  
  if (avg_time_ms < 100) {
    std::cout << "✓ 性能良好 (< 100ms)\n";
  } else if (avg_time_ms < 500) {
    std::cout << "⚠ 性能一般 (100-500ms)\n";
  } else {
    std::cout << "⚠ 性能较慢 (> 500ms)\n";
  }
}

int main(int argc, char* argv[]) {
  // 初始化 Google Logging
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  
  PrintSeparator("BERT Grammar 测试程序");
  
  // 检查命令行参数
  if (argc < 3) {
    std::cerr << "用法: " << argv[0] << " <model_path> <vocab_path> [config_path]\n";
    std::cerr << "\n示例:\n";
    std::cerr << "  " << argv[0] << " model.onnx vocab.txt\n";
    std::cerr << "  " << argv[0] << " model.onnx vocab.txt /path/to/default.yaml\n";
    return 1;
  }
  
  std::string model_path = argv[1];
  std::string vocab_path = argv[2];
  std::string config_path = (argc > 3) ? argv[3] : "";
  
  std::cout << "模型路径: " << model_path << "\n";
  std::cout << "词汇表路径: " << vocab_path << "\n";
  if (!config_path.empty()) {
    std::cout << "配置文件路径: " << config_path << "\n";
  }
  std::cout << "\n";
  
  // 创建配置对象
  Config* config = nullptr;
  if (!config_path.empty()) {
    // 从文件加载配置
    config = new Config();
    path config_file_path(config_path);
    if (!config->LoadFromFile(config_file_path)) {
      std::cerr << "错误: 无法加载配置文件: " << config_path << "\n";
      delete config;
      return 1;
    }
    // 如果配置文件中没有设置，使用命令行参数
    // 注意：如果配置文件中没有设置，我们无法使用 SetString（可能未链接）
    // 所以直接使用命令行参数创建新的配置
    string existing_model_path, existing_vocab_path;
    config->GetString("bert_grammar/model_path", &existing_model_path);
    config->GetString("bert_grammar/vocab_path", &existing_vocab_path);
    if (existing_model_path.empty() || existing_vocab_path.empty()) {
      std::cout << "警告: 配置文件中缺少模型或词汇表路径，将使用命令行参数\n";
      // 创建新的配置对象，直接使用命令行参数
      delete config;
      config = new Config();
      // 由于 SetString 可能不可用，我们创建一个临时配置文件
      std::string temp_config = "bert_grammar:\n  model_path: " + model_path + "\n  vocab_path: " + vocab_path + "\n";
      std::string temp_file = std::tmpnam(nullptr);
      temp_file += ".yaml";
      std::ofstream ofs(temp_file);
      ofs << temp_config;
      ofs.close();
      path temp_path(temp_file);
      if (!config->LoadFromFile(temp_path)) {
        std::cerr << "错误: 无法创建临时配置文件\n";
        delete config;
        return 1;
      }
      // 临时文件会在程序结束时被删除（或手动删除）
      std::remove(temp_file.c_str());
    }
  } else {
    // 创建临时配置文件
    std::string temp_config = "bert_grammar:\n  model_path: " + model_path + "\n  vocab_path: " + vocab_path + "\n";
    std::string temp_file = std::tmpnam(nullptr);
    temp_file += ".yaml";
    std::ofstream ofs(temp_file);
    ofs << temp_config;
    ofs.close();
    config = new Config();
    path temp_path(temp_file);
    if (!config->LoadFromFile(temp_path)) {
      std::cerr << "错误: 无法创建临时配置文件\n";
      delete config;
      return 1;
    }
    // 临时文件会在程序结束时被删除（或手动删除）
    std::remove(temp_file.c_str());
  }
  
  // 创建 BertGrammar 实例
  std::cout << "正在初始化 BERT Grammar 组件...\n";
  BertGrammar* grammar = new BertGrammar(config);
  
  // 运行测试
  bool all_tests_passed = true;
  
  // 测试 1: 模型加载
  if (!TestModelLoading(grammar)) {
    all_tests_passed = false;
    std::cout << "\n⚠ 模型可能未正确加载。请检查:\n";
    std::cout << "  1. 模型文件路径是否正确\n";
    std::cout << "  2. 模型文件是否存在\n";
    std::cout << "  3. ONNX Runtime 是否正确链接\n";
    std::cout << "  4. 查看上面的日志输出\n";
  }
  
  // 测试 2: 词汇表
  if (!TestVocabulary(grammar)) {
    all_tests_passed = false;
  }
  
  // 测试 3: 上下文查询
  if (!TestContextualQueries(grammar)) {
    all_tests_passed = false;
  }
  
  // 测试 4: 上下文敏感性
  if (!TestContextSensitivity(grammar)) {
    all_tests_passed = false;
  }
  
  // 测试 5: 性能
  TestPerformance(grammar);
  
  // 总结
  PrintSeparator("测试总结");
  
  if (all_tests_passed) {
    std::cout << "✓ 所有测试通过！BERT 模型正常工作。\n";
  } else {
    std::cout << "⚠ 部分测试未通过。请检查上面的输出和日志。\n";
  }
  
  std::cout << "\n提示: 查看上面的日志输出以获取更多信息。\n";
  std::cout << "如果看到 'ONNX Runtime session loaded successfully'，说明模型已加载。\n";
  std::cout << "如果看到 'BERT inference failed'，说明推理过程有问题。\n";
  
  // 清理
  delete grammar;
  delete config;
  
  return all_tests_passed ? 0 : 1;
}

