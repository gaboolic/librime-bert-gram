"""
简单的输入法流畅度评分示例
"""

from input_method_scorer_v2 import InputMethodScorer
import time
import statistics

# 初始化评分器（使用中文BERT模型，首次运行会下载模型）
# 如果网络较慢，可以先用TinyBERT: model_name='huawei-noah/TinyBERT_General_4L_312D'
print("提示：首次运行会下载bert-base-chinese模型（约400MB），请耐心等待...")
scorer = InputMethodScorer(model_name='bert-base-chinese', use_mlm_model=True)
# scorer = InputMethodScorer(model_name='ckiplab/bert-tiny-chinese', use_mlm_model=True)
# scorer = InputMethodScorer(model_name='ckiplab/albert-base-chinese', use_mlm_model=True)


# 示例：比较两个句子
sentence1 = "给磨合夹馍"
sentence2 = "给墨盒加墨"

print("句子1:", sentence1)
log_prob1 = scorer.calculate_sentence_score_mlm(sentence1)
perplexity1 = scorer.calculate_perplexity(sentence1)
coherence1 = scorer.calculate_sentence_coherence(sentence1)
combined1 = scorer.calculate_combined_score(sentence1)
print(f"对数概率: {log_prob1:.4f}")
print(f"困惑度: {perplexity1:.4f} (越低越流畅)")
print(f"连贯性: {coherence1:.4f}")
print(f"综合分数: {combined1:.4f} (越高越流畅)")

print("\n句子2:", sentence2)
log_prob2 = scorer.calculate_sentence_score_mlm(sentence2)
perplexity2 = scorer.calculate_perplexity(sentence2)
coherence2 = scorer.calculate_sentence_coherence(sentence2)
combined2 = scorer.calculate_combined_score(sentence2)
print(f"对数概率: {log_prob2:.4f}")
print(f"困惑度: {perplexity2:.4f} (越低越流畅)")
print(f"连贯性: {coherence2:.4f}")
print(f"综合分数: {combined2:.4f} (越高越流畅)")

# 使用综合分数判断（推荐）
print(f"\n判断（基于综合分数）: {'句子1更流畅' if combined1 > combined2 else '句子2更流畅'}")
print(f"判断（基于困惑度）: {'句子1更流畅' if perplexity1 < perplexity2 else '句子2更流畅'}")
print(f"判断（基于对数概率）: {'句子1更流畅' if log_prob1 > log_prob2 else '句子2更流畅'}")

# 对多个候选句子排序
candidates = [
    "粮仓里的藏书",
    "粮仓里的仓鼠",
    "图书馆里的藏书",
    "图书馆里的仓鼠",
    "预后的彩虹",
    "雨后的彩虹",
    "哈尔滨制动不再寂寞",
    "哈尔滨之冬不再寂寞",
    "北方得以中古老的",
    "北方的一种古老的",
]

print("\n" + "="*50)
print("候选句子排序（按流畅度，使用综合方法）:")
print("="*50)
ranked = scorer.rank_candidates(candidates, method='combined')
for i, sentence in enumerate(ranked, 1):
    combined = scorer.calculate_combined_score(sentence)
    perplexity = scorer.calculate_perplexity(sentence)
    log_prob = scorer.calculate_sentence_score_mlm(sentence)
    coherence = scorer.calculate_sentence_coherence(sentence)
    print(f"{i}. {sentence}")
    print(f"   综合分数: {combined:.4f}, 困惑度: {perplexity:.4f}, 对数概率: {log_prob:.4f}, 连贯性: {coherence:.4f}")

# ========== 性能测试 ==========
print("\n" + "="*60)
print("性能测试：rank_candidates 方法")
print("="*60)

# 准备测试数据
test_candidates_sets = [
    (["各个国家有各个国家的国歌", "各个国家有各个国家德国个"], "2个候选"),
    (["各个国家有各个国家的国歌", "各个国家有各个国家德国个", "各个国家有各个国家的国歌吗", 
      "冰灯是流行于中国北方的一种古老的民间艺术形式", "冰灯是流行于中国北方得以中古老的民间艺术形式"], "5个候选"),
    (["今天天气很好", "今天天气很好很", "我喜欢吃苹果", "我喜欢吃苹果的", 
      "他在看书", "他在看书书", "我们一起学习", "我们一起学习习",
      "各个国家有各个国家的国歌", "各个国家有各个国家德国个"], "10个候选"),
]

# 测试不同方法
methods = ['combined', 'mlm', 'coherence', 'perplexity']

print("\n测试配置:")
print(f"  测试轮数: 5轮（取平均值）")
print(f"  测试方法: {', '.join(methods)}")
print(f"  候选句子数量: {[f'{len(cands)}个' for cands, _ in test_candidates_sets]}")

results = {}

for candidates, label in test_candidates_sets:
    print(f"\n{'='*60}")
    print(f"测试场景: {label} ({len(candidates)}个候选句子)")
    print(f"{'='*60}")
    
    results[label] = {}
    
    for method in methods:
        # 预热（第一次运行通常较慢）
        scorer.rank_candidates(candidates[:2], method=method)
        
        # 性能测试（运行5次取平均）
        times = []
        for _ in range(5):
            start_time = time.time()
            ranked = scorer.rank_candidates(candidates, method=method)
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        throughput = len(candidates) / avg_time  # 每秒处理的句子数
        
        results[label][method] = {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_time': std_time,
            'throughput': throughput
        }
        
        print(f"\n方法: {method}")
        print(f"  平均耗时: {avg_time*1000:.2f} ms")
        print(f"  最快: {min_time*1000:.2f} ms")
        print(f"  最慢: {max_time*1000:.2f} ms")
        print(f"  标准差: {std_time*1000:.2f} ms")
        print(f"  吞吐量: {throughput:.2f} 句子/秒")

# 性能总结
print("\n" + "="*60)
print("性能总结")
print("="*60)

print("\n各方法平均耗时对比（毫秒）:")
print(f"{'方法':<15} {'2个候选':<12} {'5个候选':<12} {'10个候选':<12}")
print("-" * 60)
for method in methods:
    row = f"{method:<15}"
    for label in ["2个候选", "5个候选", "10个候选"]:
        if label in results and method in results[label]:
            avg_ms = results[label][method]['avg_time'] * 1000
            row += f"{avg_ms:>10.2f} ms  "
        else:
            row += f"{'N/A':>12}  "
    print(row)

print("\n各方法吞吐量对比（句子/秒）:")
print(f"{'方法':<15} {'2个候选':<12} {'5个候选':<12} {'10个候选':<12}")
print("-" * 60)
for method in methods:
    row = f"{method:<15}"
    for label in ["2个候选", "5个候选", "10个候选"]:
        if label in results and method in results[label]:
            throughput = results[label][method]['throughput']
            row += f"{throughput:>10.2f}     "
        else:
            row += f"{'N/A':>12}  "
    print(row)



