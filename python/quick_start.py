"""
快速开始：使用TinyBERT/BERT评估输入法句子流畅度
"""

from input_method_scorer_v2 import InputMethodScorer

# 初始化评分器
# 选项1: 使用中文BERT（推荐，准确度高，但模型较大约400MB）
scorer = InputMethodScorer(model_name='bert-base-chinese', use_mlm_model=True)

# 选项2: 使用TinyBERT（模型小，但准确度较低）
# scorer = InputMethodScorer(model_name='huawei-noah/TinyBERT_General_4L_312D', use_mlm_model=True)

# 示例：比较两个句子
sentence1 = "各个国家有各个国家的国歌"  # 正确
sentence2 = "各个国家有各个国家德国个"  # 错误

# 计算综合分数（推荐方法）
score1 = scorer.calculate_combined_score(sentence1)
score2 = scorer.calculate_combined_score(sentence2)

print(f"句子1: {sentence1}")
print(f"  综合分数: {score1:.4f} (越高越流畅)")

print(f"\n句子2: {sentence2}")
print(f"  综合分数: {score2:.4f} (越高越流畅)")

print(f"\n判断: {'句子1更流畅 ✓' if score1 > score2 else '句子2更流畅 ✗'}")

# 对多个候选句子排序
candidates = [
    "各个国家有各个国家的国歌",
    "各个国家有各个国家德国个",
    "各个国家有各个国家的国歌吗"
]

print("\n" + "="*50)
print("候选句子排序:")
print("="*50)
ranked = scorer.rank_candidates(candidates, method='combined')
for i, sentence in enumerate(ranked, 1):
    score = scorer.calculate_combined_score(sentence)
    print(f"{i}. {sentence} (分数: {score:.4f})")


