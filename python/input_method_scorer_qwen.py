"""
使用 Qwen 模型评估输入法候选句子的语法正确性和流畅度
支持多种评分方法和 Qwen 语言模型
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np


class QwenInputMethodScorer:
    """输入法句子流畅度评分器（使用 Qwen 模型）"""
    
    def __init__(self, model_name='Qwen/Qwen2-1.5B-Instruct', device=None):
        """
        初始化评分器
        
        Args:
            model_name: Qwen 模型名称
                - 'Qwen/Qwen2-1.5B-Instruct': 小模型（速度快，推荐用于测试）
                - 'Qwen/Qwen2-7B-Instruct': 中等模型（效果更好但更慢）
                - 'Qwen/Qwen2-0.5B-Instruct': 超小模型（最快但效果较差）
            device: 设备（'cuda', 'cpu' 或 None，None 时自动检测）
        """
        # 自动检测设备
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                print("未检测到 GPU，使用 CPU")
        else:
            if device == 'cuda' and not torch.cuda.is_available():
                print("警告: 指定了 CUDA 但 GPU 不可用，回退到 CPU")
                device = 'cpu'
        
        self.device = torch.device(device)
        print(f"正在加载 Qwen 模型: {model_name} (设备: {self.device})")
        print("   提示：首次运行会下载模型，请耐心等待...")
        
        # 加载 Qwen 模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        )
        
        # 将模型移动到指定设备
        self.model = self.model.to(self.device)
        self.model.eval()
        print("模型加载完成！")
    
    def calculate_perplexity(self, sentence):
        """
        计算句子的困惑度（Perplexity）
        困惑度越低，句子越流畅
        
        Args:
            sentence: 输入句子
        
        Returns:
            困惑度值（越低越好）
        """
        # 对句子进行编码
        inputs = self.tokenizer(sentence, return_tensors="pt", 
                                padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'][0].to(self.device)
        
        # 计算困惑度：使用语言模型的负对数似然
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids.unsqueeze(0))
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # 计算每个位置的对数概率
            log_probs = F.log_softmax(logits, dim=-1)
            
            # 计算每个 token 的对数概率（除了最后一个位置，因为没有下一个 token）
            total_log_prob = 0.0
            valid_tokens = 0
            
            for i in range(len(input_ids) - 1):
                token_id = input_ids[i + 1].item()
                token_log_prob = log_probs[i, token_id].item()
                total_log_prob += token_log_prob
                valid_tokens += 1
        
        if valid_tokens == 0:
            return float('inf')
        
        # 计算平均对数概率
        avg_log_prob = total_log_prob / valid_tokens
        # 困惑度 = exp(-平均对数概率)
        perplexity = np.exp(-avg_log_prob)
        return perplexity
    
    def calculate_sentence_score_mlm(self, sentence):
        """
        使用语言模型方法计算句子分数（对数概率）
        
        Args:
            sentence: 输入句子
        
        Returns:
            平均对数概率分数（分数越高越好）
        """
        # 对句子进行编码
        inputs = self.tokenizer(sentence, return_tensors="pt", 
                                padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'][0].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids.unsqueeze(0))
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # 计算每个位置的对数概率
            log_probs = F.log_softmax(logits, dim=-1)
            
            # 计算每个 token 的对数概率
            total_log_prob = 0.0
            valid_tokens = 0
            
            for i in range(len(input_ids) - 1):
                token_id = input_ids[i + 1].item()
                token_log_prob = log_probs[i, token_id].item()
                total_log_prob += token_log_prob
                valid_tokens += 1
        
        if valid_tokens == 0:
            return -float('inf')
        
        # 长度归一化：除以token数量
        avg_log_prob = total_log_prob / valid_tokens
        return avg_log_prob
    
    def calculate_sentence_coherence(self, sentence):
        """
        使用句子表示计算流畅度（基于句子嵌入的连贯性）
        这个方法通过计算句子内部token表示的相似度来评估流畅度
        
        Args:
            sentence: 输入句子
        
        Returns:
            连贯性分数（分数越高越好）
        """
        # 对句子进行编码
        inputs = self.tokenizer(sentence, return_tensors="pt", 
                                padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'][0].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids.unsqueeze(0), output_hidden_states=True)
            
            # 获取最后一层的隐藏状态
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1][0]  # [seq_len, hidden_size]
            else:
                # 如果没有 hidden_states，尝试从 logits 获取（需要访问模型的嵌入层）
                # 这里使用一个简化的方法：使用 logits 的归一化作为近似
                logits = outputs.logits[0]  # [seq_len, vocab_size]
                # 使用 PCA 或简单的降维方法（这里简化处理）
                hidden_states = F.normalize(logits, p=2, dim=-1)
        
        # 计算相邻token之间的相似度
        if len(hidden_states) < 2:
            return 0.0
        
        # 只考虑有效token（跳过特殊token）
        valid_indices = list(range(1, len(hidden_states) - 1))  # 跳过第一个和最后一个
        
        if len(valid_indices) < 2:
            return 0.0
        
        # 计算相邻token的余弦相似度
        similarities = []
        for i in range(len(valid_indices) - 1):
            idx1, idx2 = valid_indices[i], valid_indices[i + 1]
            emb1 = hidden_states[idx1]
            emb2 = hidden_states[idx2]
            # 余弦相似度
            cos_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
            similarities.append(cos_sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def calculate_combined_score(self, sentence, mlm_weight=0.7, coherence_weight=0.3):
        """
        组合多种方法计算综合分数
        
        Args:
            sentence: 输入句子
            mlm_weight: MLM分数的权重
            coherence_weight: 连贯性分数的权重
        
        Returns:
            综合分数（分数越高越好）
        """
        # 归一化MLM分数（转换为0-1范围）
        mlm_score = self.calculate_sentence_score_mlm(sentence)
        # 使用sigmoid将分数映射到0-1范围
        mlm_normalized = 1 / (1 + np.exp(-mlm_score / 2))  # 除以2是为了调整sigmoid的陡峭度
        
        # 连贯性分数已经在0-1范围
        coherence_score = self.calculate_sentence_coherence(sentence)
        
        # 组合分数
        combined = mlm_weight * mlm_normalized + coherence_weight * coherence_score
        
        return combined
    
    def compare_sentences(self, sentences, method='combined'):
        """
        比较多个句子的流畅度
        
        Args:
            sentences: 句子列表
            method: 计算方法
                - 'mlm': 使用MLM方法（对数概率）
                - 'perplexity': 使用困惑度
                - 'coherence': 使用连贯性
                - 'combined': 组合方法（推荐）
        
        Returns:
            排序后的句子和分数列表，按流畅度从高到低排序
        """
        results = []
        
        for sentence in sentences:
            if method == 'mlm':
                score = self.calculate_sentence_score_mlm(sentence)
                sort_score = score
            elif method == 'perplexity':
                score = self.calculate_perplexity(sentence)
                sort_score = -score  # 困惑度越低越好
            elif method == 'coherence':
                score = self.calculate_sentence_coherence(sentence)
                sort_score = score
            elif method == 'combined':
                score = self.calculate_combined_score(sentence)
                sort_score = score
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results.append({
                'sentence': sentence,
                'score': score,
                'sort_score': sort_score
            })
        
        # 按sort_score从高到低排序（最流畅的在前）
        results.sort(key=lambda x: x['sort_score'], reverse=True)
        
        return results
    
    def rank_candidates(self, candidates, method='combined'):
        """
        对输入法候选句子进行排序
        
        Args:
            candidates: 候选句子列表
            method: 排序方法 ('mlm', 'perplexity', 'coherence', 'combined')
        
        Returns:
            排序后的候选句子列表（最流畅的在前）
        """
        results = self.compare_sentences(candidates, method=method)
        return [r['sentence'] for r in results]


def main():
    """主函数：演示输入法场景的使用"""
    
    # 初始化评分器（使用 Qwen 模型）
    print("=" * 60)
    print("初始化输入法流畅度评分器（使用 Qwen 模型）")
    print("=" * 60)
    # 自动检测并使用 GPU（如果可用）
    scorer = QwenInputMethodScorer(model_name='Qwen/Qwen2-1.5B-Instruct', device=None)
    
    # 示例1：用户提供的例子
    print("\n" + "=" * 60)
    print("示例1: 判断句子流畅度（使用组合方法）")
    print("=" * 60)
    
    sentences = [
        "各个国家有各个国家的国歌",
        "各个国家有各个国家德国个"
    ]
    
    print("\n候选句子:")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")
    
    print("\n评分结果（使用组合方法）:")
    results = scorer.compare_sentences(sentences, method='combined')
    for i, result in enumerate(results, 1):
        sentence = result['sentence']
        combined_score = result['score']
        mlm_score = scorer.calculate_sentence_score_mlm(sentence)
        coherence_score = scorer.calculate_sentence_coherence(sentence)
        perplexity = scorer.calculate_perplexity(sentence)
        
        print(f"\n  排名 {i}: {sentence}")
        print(f"    综合分数: {combined_score:.4f}")
        print(f"    MLM对数概率: {mlm_score:.4f}")
        print(f"    连贯性分数: {coherence_score:.4f}")
        print(f"    困惑度: {perplexity:.4f}")
    
    # 示例2：多个候选句子排序
    print("\n" + "=" * 60)
    print("示例2: 输入法候选句子排序")
    print("=" * 60)
    
    candidates = [
        "各个国家有各个国家的国歌",
        "各个国家有各个国家德国个",
        "各个国家有各个国家的国歌吗"
    ]
    
    print("\n候选句子:")
    for i, cand in enumerate(candidates, 1):
        print(f"  {i}. {cand}")
    
    print("\n排序结果（按流畅度从高到低，使用组合方法）:")
    ranked = scorer.rank_candidates(candidates, method='combined')
    for i, sentence in enumerate(ranked, 1):
        combined_score = scorer.calculate_combined_score(sentence)
        print(f"  {i}. {sentence} (综合分数: {combined_score:.4f})")


if __name__ == "__main__":
    main()

