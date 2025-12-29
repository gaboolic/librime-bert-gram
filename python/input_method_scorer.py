"""
使用TinyBERT评估输入法候选句子的语法正确性和流畅度
用于判断哪个句子更符合语法、更流畅
"""

from transformers import BertForMaskedLM, BertTokenizer
import torch
import torch.nn.functional as F
import numpy as np


class InputMethodScorer:
    """输入法句子流畅度评分器"""
    
    def __init__(self, model_name='huawei-noah/TinyBERT_General_4L_312D'):
        """
        初始化评分器
        
        Args:
            model_name: TinyBERT模型名称
        """
        print(f"正在加载模型: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        print("模型加载完成！")
    
    def calculate_sentence_score(self, sentence, method='avg_log_prob'):
        """
        计算句子的流畅度分数（使用掩码语言模型方法）
        
        Args:
            sentence: 输入句子
            method: 计算方法（已弃用，统一使用MLM方法）
        
        Returns:
            分数值（平均对数概率，分数越高越好）
        """
        # 直接使用MLM方法，更准确
        return self.calculate_sentence_score_mlm(sentence)
    
    def calculate_sentence_score_mlm(self, sentence):
        """
        使用掩码语言模型方法计算句子分数
        通过掩码每个位置并计算预测概率
        
        Args:
            sentence: 输入句子
        
        Returns:
            平均对数概率分数（分数越高越好）
        """
        # 对句子进行编码
        inputs = self.tokenizer(sentence, return_tensors="pt", 
                                padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        
        total_log_prob = 0.0
        valid_tokens = 0
        
        # 对每个位置进行掩码并计算概率
        for i in range(1, len(input_ids) - 1):  # 跳过[CLS]和[SEP]
            if attention_mask[i] == 0:
                continue
            
            # 创建掩码输入
            masked_input_ids = input_ids.clone()
            original_token_id = input_ids[i].item()
            masked_input_ids[i] = self.tokenizer.mask_token_id
            
            # 获取预测（使用unsqueeze添加batch维度）
            masked_input_ids_batch = masked_input_ids.unsqueeze(0)
            attention_mask_batch = attention_mask.unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(input_ids=masked_input_ids_batch, 
                                    attention_mask=attention_mask_batch)
                logits = outputs.logits
                # 使用log_softmax更稳定，直接获取对数概率
                log_probs = F.log_softmax(logits[0, i], dim=-1)
                token_log_prob = log_probs[original_token_id].item()
            
            total_log_prob += token_log_prob
            valid_tokens += 1
        
        if valid_tokens == 0:
            return -float('inf')
        
        return total_log_prob / valid_tokens
    
    def calculate_perplexity(self, sentence):
        """
        计算句子的困惑度（Perplexity）
        困惑度越低，句子越流畅
        
        Args:
            sentence: 输入句子
        
        Returns:
            困惑度值（越低越好）
        """
        avg_log_prob = self.calculate_sentence_score_mlm(sentence)
        if avg_log_prob == -float('inf'):
            return float('inf')
        perplexity = np.exp(-avg_log_prob)
        return perplexity
    
    def compare_sentences(self, sentences, method='mlm', use_perplexity=False):
        """
        比较多个句子的流畅度
        
        Args:
            sentences: 句子列表
            method: 计算方法 ('mlm' 或 'avg_log_prob')
            use_perplexity: 是否使用困惑度（True时困惑度越低越好，False时对数概率越高越好）
        
        Returns:
            排序后的句子和分数列表，按流畅度从高到低排序
        """
        results = []
        
        for sentence in sentences:
            if use_perplexity:
                score = self.calculate_perplexity(sentence)
                # 困惑度越低越好，所以排序时取负值
                sort_score = -score
            else:
                if method == 'mlm':
                    score = self.calculate_sentence_score_mlm(sentence)
                else:
                    score = self.calculate_sentence_score(sentence, method=method)
                sort_score = score
            
            results.append({
                'sentence': sentence,
                'score': score,
                'sort_score': sort_score
            })
        
        # 按sort_score从高到低排序（最流畅的在前）
        results.sort(key=lambda x: x['sort_score'], reverse=True)
        
        return results
    
    def rank_candidates(self, candidates, use_perplexity=False):
        """
        对输入法候选句子进行排序
        
        Args:
            candidates: 候选句子列表
            use_perplexity: 是否使用困惑度进行排序
        
        Returns:
            排序后的候选句子列表（最流畅的在前）
        """
        results = self.compare_sentences(candidates, method='mlm', use_perplexity=use_perplexity)
        return [r['sentence'] for r in results]


def main():
    """主函数：演示输入法场景的使用"""
    
    # 初始化评分器
    print("=" * 60)
    print("初始化输入法流畅度评分器")
    print("=" * 60)
    scorer = InputMethodScorer()
    
    # 示例1：用户提供的例子
    print("\n" + "=" * 60)
    print("示例1: 判断句子流畅度")
    print("=" * 60)
    
    sentences = [
        "各个国家有各个国家的国歌",
        "各个国家有各个国家德国个"
    ]
    
    print("\n候选句子:")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")
    
    print("\n评分结果（使用困惑度，越低越流畅）:")
    results = scorer.compare_sentences(sentences, method='mlm', use_perplexity=True)
    for i, result in enumerate(results, 1):
        perplexity = result['score']
        log_prob = scorer.calculate_sentence_score_mlm(result['sentence'])
        print(f"  排名 {i}: {result['sentence']}")
        print(f"    困惑度: {perplexity:.4f} (越低越流畅)")
        print(f"    对数概率: {log_prob:.4f} (越高越流畅)")
    
    # 示例2：多个候选句子排序
    print("\n" + "=" * 60)
    print("示例2: 输入法候选句子排序")
    print("=" * 60)
    
    candidates = [
        "我今天要去学校",
        "我今天要去学校上课",
        "我今天要去学校学习",
        "我今天要去学校上课学习",
        "我今天要去学校上课学习知识"
    ]
    
    print("\n候选句子:")
    for i, cand in enumerate(candidates, 1):
        print(f"  {i}. {cand}")
    
    print("\n排序结果（按流畅度从高到低，使用困惑度）:")
    ranked = scorer.rank_candidates(candidates, use_perplexity=True)
    for i, sentence in enumerate(ranked, 1):
        perplexity = scorer.calculate_perplexity(sentence)
        log_prob = scorer.calculate_sentence_score_mlm(sentence)
        print(f"  {i}. {sentence}")
        print(f"     困惑度: {perplexity:.4f}, 对数概率: {log_prob:.4f}")
    
    # 示例3：语法错误检测
    print("\n" + "=" * 60)
    print("示例3: 语法错误检测")
    print("=" * 60)
    
    test_pairs = [
        ("我喜欢吃苹果", "我喜欢吃苹果的"),
        ("今天天气很好", "今天天气很好很"),
        ("他在看书", "他在看书书"),
        ("我们一起学习", "我们一起学习习")
    ]
    
    print("\n对比结果（使用困惑度）:")
    for correct, wrong in test_pairs:
        perplexity_correct = scorer.calculate_perplexity(correct)
        perplexity_wrong = scorer.calculate_perplexity(wrong)
        log_prob_correct = scorer.calculate_sentence_score_mlm(correct)
        log_prob_wrong = scorer.calculate_sentence_score_mlm(wrong)
        
        print(f"\n  正确: {correct}")
        print(f"    困惑度: {perplexity_correct:.4f}, 对数概率: {log_prob_correct:.4f}")
        print(f"  错误: {wrong}")
        print(f"    困惑度: {perplexity_wrong:.4f}, 对数概率: {log_prob_wrong:.4f}")
        
        # 使用困惑度判断（困惑度越低越好）
        is_correct = perplexity_correct < perplexity_wrong
        print(f"  判断: {'✓ 正确识别' if is_correct else '✗ 判断错误'}")


if __name__ == "__main__":
    main()

