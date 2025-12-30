"""
结合 librime 和 BERT 模型的整句输入法演示

功能：
1. 使用 librime 输入整句拼音，获取候选词
2. 使用 BERT 模型对候选词进行流畅度评分
3. 根据流畅度重新排序，选择最流畅的句子
"""

import os
import time
from call_librime import RimeDllWrapper
from input_method_scorer_v2 import InputMethodScorer
from typing import List, Dict, Tuple, Optional
from itertools import product


class RimeBertInputMethod:
    """结合 Rime 和 BERT 的输入法"""
    
    def __init__(self, rime_dll_path=None, bert_model_name='bert-base-chinese', use_mlm_model=True):
        """
        初始化输入法
        
        Args:
            rime_dll_path: rime.dll 的路径，如果为 None 会自动查找
            bert_model_name: BERT 模型名称
            use_mlm_model: 是否使用 MLM 模型
        """
        print("=" * 70)
        print("初始化 Rime + BERT 输入法")
        print("=" * 70)
        
        # 初始化 Rime
        print("\n[1/2] 初始化 Rime...")
        self.rime = RimeDllWrapper(dll_path=rime_dll_path)
        print(f"   ✓ 找到 rime.dll: {self.rime.dll_path}")
        
        # 设置数据目录
        dll_dir = os.path.dirname(self.rime.dll_path)  # build/bin/Release
        data_dir = os.path.dirname(dll_dir)  # build/bin
        
        print(f"   DLL 目录: {dll_dir}")
        print(f"   数据目录: {data_dir}")
        
        # 初始化 Rime
        self.rime.initialize(
            app_name="rime.bert.python",
            shared_data_dir=data_dir,
            user_data_dir=data_dir
        )
        print("   ✓ Rime 初始化完成")
        
        # 初始化 BERT 评分器
        print(f"\n[2/2] 初始化 BERT 评分器（模型: {bert_model_name}）...")
        print("   提示：首次运行会下载模型，请耐心等待...")
        self.scorer = InputMethodScorer(
            model_name=bert_model_name,
            use_mlm_model=use_mlm_model
        )
        print("   ✓ BERT 评分器初始化完成")
        
        self.session_id = None
        print("\n" + "=" * 70)
        print("初始化完成！")
        print("=" * 70)
    
    def create_session(self, schema_name="luna_pinyin"):
        """
        创建输入会话
        
        Args:
            schema_name: 输入方案名称（默认：luna_pinyin）
        """
        if self.session_id:
            self.destroy_session()
        
        self.session_id = self.rime.create_session()
        if self.session_id:
            # 选择输入方案
            current_schema = self.rime.get_current_schema(self.session_id)
            if current_schema != schema_name:
                if self.rime.select_schema(self.session_id, schema_name):
                    print(f"   ✓ 已选择输入方案: {schema_name}")
                else:
                    print(f"   ⚠ 选择方案失败，当前方案: {current_schema}")
            return True
        else:
            print("   ✗ 会话创建失败")
            return False
    
    def destroy_session(self):
        """销毁当前会话"""
        if self.session_id:
            self.rime.destroy_session(self.session_id)
            self.session_id = None
    
    def clear_input(self):
        """
        清除当前输入
        
        方法：销毁并重新创建会话，这样可以确保输入状态完全清除
        如果会话不存在，则创建新会话
        """
        # 如果会话不存在，直接创建新会话
        if not self.session_id:
            return self.create_session()
        
        # 在销毁前获取当前方案
        schema_name = self.rime.get_current_schema(self.session_id) or "luna_pinyin"
        
        # 销毁并重新创建会话，确保输入状态完全清除
        self.destroy_session()
        return self.create_session(schema_name)
    
    def input_pinyin(self, pinyin_text):
        """
        输入拼音并获取候选词（使用 BERT 评分排序）
        
        Args:
            pinyin_text: 拼音字符串（全小写，无空格，例如："tushuguanlidecangshu"）
        
        Returns:
            dict: 包含排序后的候选词和评分信息
        """
        # 0. 清除之前的输入（确保每次输入都是全新的）
        # 如果会话不存在，clear_input 会创建新会话
        if not self.clear_input():
            return None
        
        print(f"\n输入拼音: {pinyin_text}")
        
        # 1. 输入拼音到 Rime
        if not self.rime.simulate_key_sequence(self.session_id, pinyin_text):
            print("   ✗ 输入失败")
            return None
        
        # 2. 获取候选词
        context = self.rime.get_context(self.session_id)
        if not context:
            print("   ✗ 无法获取上下文")
            return None
        
        candidates = context.get('candidates', [])
        if not candidates:
            print("   ⚠ 未获取到候选词")
            return None
        
        # 提取候选词文本
        candidate_texts = [cand.get('text', '') for cand in candidates if cand.get('text', '')]
        
        if not candidate_texts:
            print("   ⚠ 候选词为空")
            return None
        
        print(f"   ✓ 获取到 {len(candidate_texts)} 个候选词")
        
        # 3. 使用 BERT 评分器对候选词进行评分和排序
        print(f"\n使用 BERT 模型对候选词进行流畅度评分...")
        start_time = time.time()
        
        # 使用综合方法排序（推荐）
        ranked_results = self.scorer.compare_sentences(candidate_texts, method='combined')
        
        elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒
        print(f"   ✓ 评分完成（耗时: {elapsed_time:.2f} ms）")
        
        # 4. 构建结果
        result = {
            'input': context.get('input', ''),
            'original_candidates': candidates,
            'ranked_candidates': [],
            'scoring_time_ms': elapsed_time
        }
        
        # 添加排序后的候选词和评分信息
        for i, ranked_item in enumerate(ranked_results):
            sentence = ranked_item['sentence']
            combined_score = ranked_item['score']
            
            # 计算其他评分指标（用于显示）
            mlm_score = self.scorer.calculate_sentence_score_mlm(sentence)
            coherence_score = self.scorer.calculate_sentence_coherence(sentence)
            perplexity = self.scorer.calculate_perplexity(sentence)
            
            # 找到原始候选词信息
            original_cand = next(
                (c for c in candidates if c.get('text', '') == sentence),
                {'text': sentence, 'comment': ''}
            )
            
            result['ranked_candidates'].append({
                'rank': i + 1,
                'text': sentence,
                'comment': original_cand.get('comment', ''),
                'scores': {
                    'combined': combined_score,
                    'mlm': mlm_score,
                    'coherence': coherence_score,
                    'perplexity': perplexity
                }
            })
        
        return result
    
    def get_candidates_for_pinyin(self, pinyin_text: str, max_candidates: int = 10) -> List[str]:
        """
        获取指定拼音的候选词列表
        
        Args:
            pinyin_text: 拼音字符串
            max_candidates: 最多返回的候选词数量
        
        Returns:
            候选词文本列表
        """
        if not self.session_id:
            if not self.create_session():
                return []
        
        # 重要：每次获取候选词前，先清除之前的输入
        # 保存当前方案
        schema_name = self.rime.get_current_schema(self.session_id) or "luna_pinyin"
        self.destroy_session()
        if not self.create_session(schema_name):
            return []
        
        # 输入拼音
        if not self.rime.simulate_key_sequence(self.session_id, pinyin_text):
            return []
        
        # 获取候选词
        context = self.rime.get_context(self.session_id)
        if not context:
            return []
        
        candidates = context.get('candidates', [])
        candidate_texts = [cand.get('text', '') for cand in candidates[:max_candidates] if cand.get('text', '')]
        
        return candidate_texts
    
    def input_sentence_pinyin(self, pinyin_text: str, max_segment_candidates: int = 5, 
                             max_combinations: int = 100) -> Optional[Dict]:
        """
        整句输入：输入整句拼音，优先使用 Rime 的完整候选词，用 BERT 评估完整句子
        
        Args:
            pinyin_text: 整句拼音（全小写，无空格，例如："congmingdeshurufa"）
            max_segment_candidates: 每个分段最多考虑的候选词数量（仅在分段模式下使用）
            max_combinations: 最多评估的组合数量（避免组合爆炸）
        
        Returns:
            包含最佳完整句子和评分信息的字典
        """
        if not self.clear_input():
            return None
        
        print(f"\n输入整句拼音: {pinyin_text}")
        
        # 1. 输入整句拼音到 Rime，让 Rime 自动切分音节
        if not self.rime.simulate_key_sequence(self.session_id, pinyin_text):
            print("   ✗ 输入失败")
            return None
        
        # 2. 获取 Rime 的上下文，它会包含切分后的拼音和完整候选词
        context = self.rime.get_context(self.session_id)
        if not context:
            print("   ✗ 无法获取上下文")
            return None
        
        # 3. 从 Rime 的输入文本中提取音节切分
        input_text = context.get('input', '')
        if not input_text:
            print("   ✗ 无法获取切分后的拼音")
            return None
        
        print(f"   Rime 切分结果: {input_text}")
        
        # 4. 优先使用 Rime 直接返回的完整候选词
        candidates = context.get('candidates', [])
        rime_full_candidates = [cand.get('text', '') for cand in candidates if cand.get('text', '')]
        
        if rime_full_candidates:
            print(f"   ✓ Rime 返回了 {len(rime_full_candidates)} 个完整候选词")
            print(f"   前5个候选词: {', '.join(rime_full_candidates[:5])}")
            
            # 如果 Rime 返回的候选词足够多，直接使用它们
            all_combinations = rime_full_candidates[:max_combinations]
            print(f"\n使用 Rime 的完整候选词（共 {len(all_combinations)} 个）...")
            
            # 使用 BERT 评估这些完整候选词
            print(f"\n使用 BERT 模型评估 {len(all_combinations)} 个完整句子...")
            start_time = time.time()
            
            ranked_results = self.scorer.compare_sentences(all_combinations, method='combined')
            
            elapsed_time = (time.time() - start_time) * 1000
            print(f"   ✓ 评估完成（耗时: {elapsed_time:.2f} ms）")
            
            # 构建结果
            result = {
                'input_pinyin': pinyin_text,
                'rime_input': input_text,
                'syllables': input_text.strip().split(),
                'segments': [],  # 使用完整候选词时不需要分段信息
                'segment_candidates': [],
                'total_combinations': len(all_combinations),
                'ranked_sentences': [],
                'scoring_time_ms': elapsed_time,
                'method': 'rime_full_candidates'  # 标记使用的方法
            }
            
            # 添加排序后的完整句子
            for i, ranked_item in enumerate(ranked_results[:20]):  # 只保存前20个
                sentence = ranked_item['sentence']
                combined_score = ranked_item['score']
                
                mlm_score = self.scorer.calculate_sentence_score_mlm(sentence)
                coherence_score = self.scorer.calculate_sentence_coherence(sentence)
                perplexity = self.scorer.calculate_perplexity(sentence)
                
                result['ranked_sentences'].append({
                    'rank': i + 1,
                    'sentence': sentence,
                    'scores': {
                        'combined': combined_score,
                        'mlm': mlm_score,
                        'coherence': coherence_score,
                        'perplexity': perplexity
                    }
                })
            
            return result
        
        # 5. 如果 Rime 没有返回完整候选词，则使用分段组合的方法（备用方案）
        print("   ⚠ Rime 未返回完整候选词，使用分段组合方法...")
        
        # 解析音节切分（按空格分割）
        syllables = input_text.strip().split()
        if not syllables:
            print("   ✗ 无法解析音节")
            return None
        
        print(f"   音节数量: {len(syllables)}")
        print(f"   音节列表: {syllables}")
        
        # 为每个音节段获取候选词
        # 使用动态规划找到最佳分段方案
        print("\n正在获取各音节段的候选词...")
        
        # 使用动态规划：dp[i] 表示从位置 i 开始的最佳分段方案
        # 存储格式: (分段列表, 评分)
        dp = {}
        
        def find_best_segmentation(start: int) -> Tuple[List[Dict], float]:
            """递归查找从 start 位置开始的最佳分段"""
            if start >= len(syllables):
                return [], 0.0
            
            if start in dp:
                return dp[start]
            
            best_segments = []
            best_score = -float('inf')
            
            # 尝试不同的分段长度（优先考虑常见的中文词长度：2-3个音节）
            # 常见长度：2(双字词), 3(三字词), 4(四字词如"各个国家"), 1(单字), 5(五字词)
            lengths_to_try = [2, 3, 4, 1, 5]  # 优先尝试2-4个音节
            
            for length in lengths_to_try:
                if start + length > len(syllables):
                    continue
                
                segment_syllables = syllables[start:start+length]
                segment_pinyin = ''.join(segment_syllables)  # 无空格连接
                
                # 检查这个分段是否有候选词
                candidates = self.get_candidates_for_pinyin(segment_pinyin, max_segment_candidates)
                if not candidates:
                    continue
                
                # 计算这个分段的评分
                # 评分规则：
                # 1. 候选词数量越多，得分越高
                # 2. 2-3个音节的词得分更高（更常见）
                # 3. 4-5个音节的词（如"各个国家"）也有加分
                # 4. 1个音节的词得分较低，但如果是常见单字（如"有"、"的"）应该允许
                candidate_score = len(candidates) * 0.1
                length_bonus = 0.0
                if length == 2 or length == 3:
                    length_bonus = 0.5  # 2-3个音节的词更常见，加分
                elif length == 4 or length == 5:
                    length_bonus = 0.3  # 4-5个音节的词（如"各个国家"）也有加分
                elif length == 1:
                    # 单字词：检查是否是常见单字
                    common_single_chars = {'you', 'de', 'le', 'ma', 'ne', 'ba', 'a', 'o', 'e'}
                    if segment_pinyin in common_single_chars:
                        length_bonus = 0.2  # 常见单字有加分
                    else:
                        length_bonus = -0.3  # 其他单字词得分较低
                
                segment_score = candidate_score + length_bonus
                
                # 递归处理剩余部分
                remaining_segments, remaining_score = find_best_segmentation(start + length)
                
                # 总评分
                total_score = segment_score + remaining_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_segments = [{
                        'pinyin': segment_pinyin,
                        'syllables': segment_syllables,
                        'candidates': candidates
                    }] + remaining_segments
            
            # 如果没有找到有效分段，使用单个音节（作为后备）
            if not best_segments and start < len(syllables):
                single_pinyin = syllables[start]
                single_candidates = self.get_candidates_for_pinyin(single_pinyin, max_segment_candidates)
                remaining_segments, _ = find_best_segmentation(start + 1)
                best_segments = [{
                    'pinyin': single_pinyin,
                    'syllables': [syllables[start]],
                    'candidates': single_candidates if single_candidates else [single_pinyin]
                }] + remaining_segments
                best_score = -1.0  # 单字符分段得分较低
            
            dp[start] = (best_segments, best_score)
            return best_segments, best_score
        
        # 从位置 0 开始查找最佳分段
        segment_candidates, _ = find_best_segmentation(0)
        
        # 打印分段信息和候选词（用于调试）
        for i, seg in enumerate(segment_candidates):
            syllables_str = ' '.join(seg['syllables'])
            top_candidates = seg['candidates'][:3]  # 显示前3个候选词
            candidates_str = ', '.join(top_candidates) if top_candidates else '无'
            print(f"   分段 {i+1} ({seg['pinyin']}, {len(seg['syllables'])}个音节): {len(seg['candidates'])} 个候选词")
            if top_candidates:
                print(f"      候选词示例: {candidates_str}")
        
        if not segment_candidates:
            print("   ✗ 所有分段都没有候选词")
            return None
        
        # 3. 生成所有可能的组合
        print("\n正在生成组合...")
        all_combinations = []
        candidate_lists = [seg['candidates'] for seg in segment_candidates]
        
        # 限制组合数量，避免组合爆炸
        total_combinations = 1
        for cand_list in candidate_lists:
            total_combinations *= len(cand_list)
            if total_combinations > max_combinations:
                # 如果组合太多，只取每个分段的前几个候选词
                limited_lists = []
                for cand_list in candidate_lists:
                    limited_lists.append(cand_list[:max(1, max_combinations // len(candidate_lists))])
                candidate_lists = limited_lists
                break
        
        # 生成所有组合
        for combination in product(*candidate_lists):
            sentence = ''.join(combination)
            all_combinations.append(sentence)
            if len(all_combinations) >= max_combinations:
                break
        
        print(f"   ✓ 生成 {len(all_combinations)} 个组合")
        
        # 4. 输出所有生成的组合（用于调试）
        print(f"\n生成的完整句子组合（前20个）:")
        for i, combo in enumerate(all_combinations[:20], 1):
            print(f"   {i}. {combo}")
        if len(all_combinations) > 20:
            print(f"   ... 还有 {len(all_combinations) - 20} 个组合")
        
        # 5. 使用 BERT 评估所有组合
        print(f"\n使用 BERT 模型评估 {len(all_combinations)} 个完整句子...")
        start_time = time.time()
        
        ranked_results = self.scorer.compare_sentences(all_combinations, method='combined')
        
        elapsed_time = (time.time() - start_time) * 1000
        print(f"   ✓ 评估完成（耗时: {elapsed_time:.2f} ms）")
        
        # 6. 构建结果
        # 将 segment_candidates 转换为 segments 格式用于显示
        segments = [{'pinyin': seg['pinyin'], 'syllables': seg['syllables']} for seg in segment_candidates]
        
        result = {
            'input_pinyin': pinyin_text,
            'rime_input': input_text,  # Rime 切分后的拼音
            'syllables': syllables,  # 音节列表
            'segments': segments,  # 分段信息
            'segment_candidates': segment_candidates,
            'total_combinations': len(all_combinations),
            'ranked_sentences': [],
            'scoring_time_ms': elapsed_time,
            'method': 'segment_combination'  # 标记使用的方法
        }
        
        # 添加排序后的完整句子
        for i, ranked_item in enumerate(ranked_results[:20]):  # 只保存前20个
            sentence = ranked_item['sentence']
            combined_score = ranked_item['score']
            
            mlm_score = self.scorer.calculate_sentence_score_mlm(sentence)
            coherence_score = self.scorer.calculate_sentence_coherence(sentence)
            perplexity = self.scorer.calculate_perplexity(sentence)
            
            result['ranked_sentences'].append({
                'rank': i + 1,
                'sentence': sentence,
                'scores': {
                    'combined': combined_score,
                    'mlm': mlm_score,
                    'coherence': coherence_score,
                    'perplexity': perplexity
                }
            })
        
        return result
    
    def _segment_pinyin(self, pinyin_text: str, max_candidates_per_segment: int = 5) -> List[Dict]:
        """
        将拼音文本分段（使用动态规划找到最佳分段）
        
        策略：尝试所有可能的分段点，选择候选词质量最好的分段方案
        
        Args:
            pinyin_text: 拼音文本
            max_candidates_per_segment: 每个分段最多考虑的候选词数量（用于判断分段是否有效）
        
        Returns:
            分段列表，每个分段包含 {'pinyin': '...', 'start': 0, 'end': 5}
        """
        n = len(pinyin_text)
        if n == 0:
            return []
        
        # 使用动态规划找到最佳分段
        # dp[i] 表示从位置 i 开始的最佳分段方案和评分
        # 存储格式: (分段列表, 评分)
        dp = {}
        
        def find_best_segmentation(start: int) -> Tuple[List[Dict], float]:
            """递归查找从 start 位置开始的最佳分段"""
            if start >= n:
                return [], 0.0
            
            if start in dp:
                return dp[start]
            
            best_segments = []
            best_score = -float('inf')
            
            # 尝试不同的分段长度（优先考虑常见的中文拼音长度）
            # 常见长度：9(三字词如"图书馆"), 6(三字词), 5(双字词+单字), 4(双字词), 7(三字词+单字), 8(四字词), 3(单字+单字), 2(单字)
            lengths_to_try = [9, 6, 5, 4, 7, 8, 10, 3, 2]
            
            for length in lengths_to_try:
                if start + length > n:
                    continue
                
                segment_pinyin = pinyin_text[start:start+length]
                
                # 检查这个分段是否有候选词（只检查前3个，快速判断）
                candidates = self.get_candidates_for_pinyin(segment_pinyin, 3)
                if not candidates:
                    continue
                
                # 计算这个分段的评分
                # 有候选词的分段得分更高，候选词越多得分越高
                # 同时，更长的有效分段得分也更高
                segment_score = len(candidates) * 0.1 + length * 0.05
                
                # 递归处理剩余部分
                remaining_segments, remaining_score = find_best_segmentation(start + length)
                
                # 总评分
                total_score = segment_score + remaining_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_segments = [{
                        'pinyin': segment_pinyin,
                        'start': start,
                        'end': start + length
                    }] + remaining_segments
            
            # 如果没有找到有效分段，尝试单个字符（作为后备）
            if not best_segments and start < n:
                single_char_segment = {
                    'pinyin': pinyin_text[start],
                    'start': start,
                    'end': start + 1
                }
                remaining_segments, _ = find_best_segmentation(start + 1)
                best_segments = [single_char_segment] + remaining_segments
                best_score = -1.0  # 单字符分段得分较低
            
            dp[start] = (best_segments, best_score)
            return best_segments, best_score
        
        # 从位置 0 开始查找最佳分段
        segments, _ = find_best_segmentation(0)
        
        return segments
    
    def display_sentence_results(self, result: Dict, top_n: int = 10):
        """
        显示整句输入的结果
        
        Args:
            result: input_sentence_pinyin 返回的结果字典
            top_n: 显示前 N 个完整句子
        """
        if not result:
            return
        
        print("\n" + "=" * 70)
        print("整句输入结果（按流畅度排序）")
        print("=" * 70)
        print(f"输入拼音: {result['input_pinyin']}")
        print(f"分段数量: {len(result['segments'])}")
        print(f"生成组合: {result['total_combinations']} 个")
        print(f"评估耗时: {result['scoring_time_ms']:.2f} ms")
        
        print(f"\n分段信息:")
        for i, seg in enumerate(result['segments']):
            print(f"  分段 {i+1}: {seg['pinyin']}")
        
        print(f"\n前 {min(top_n, len(result['ranked_sentences']))} 个完整句子：")
        print("-" * 70)
        print(f"{'排名':<6} {'完整句子':<40} {'综合分数':<12} {'困惑度':<12} {'连贯性':<10}")
        print("-" * 70)
        
        for item in result['ranked_sentences'][:top_n]:
            sentence = item['sentence']
            if len(sentence) > 38:
                sentence = sentence[:35] + "..."
            
            scores = item['scores']
            print(f"{item['rank']:<6} {sentence:<40} "
                  f"{scores['combined']:<12.4f} {scores['perplexity']:<12.4f} "
                  f"{scores['coherence']:<10.4f}")
        
        # 显示最佳完整句子
        if result['ranked_sentences']:
            best = result['ranked_sentences'][0]
            print("\n" + "=" * 70)
            print(f"✨ 最佳完整句子（最流畅）: {best['sentence']}")
            print(f"   综合分数: {best['scores']['combined']:.4f}")
            print(f"   困惑度: {best['scores']['perplexity']:.4f} (越低越好)")
            print(f"   连贯性: {best['scores']['coherence']:.4f}")
            print("=" * 70)
    
    def display_results(self, result, top_n=10):
        """
        显示评分结果
        
        Args:
            result: input_pinyin 返回的结果字典
            top_n: 显示前 N 个候选词
        """
        if not result:
            return
        
        print("\n" + "=" * 70)
        print("评分结果（按流畅度排序）")
        print("=" * 70)
        print(f"输入文本: {result['input']}")
        print(f"评分耗时: {result['scoring_time_ms']:.2f} ms")
        print(f"\n前 {min(top_n, len(result['ranked_candidates']))} 个候选词：")
        print("-" * 70)
        print(f"{'排名':<6} {'候选词':<30} {'综合分数':<12} {'困惑度':<12} {'连贯性':<10}")
        print("-" * 70)
        
        for cand in result['ranked_candidates'][:top_n]:
            text = cand['text']
            if len(text) > 28:
                text = text[:25] + "..."
            
            comment = cand['comment']
            if comment and len(comment) > 0:
                text_with_comment = f"{text} ({comment})"
            else:
                text_with_comment = text
            
            scores = cand['scores']
            print(f"{cand['rank']:<6} {text_with_comment:<30} "
                  f"{scores['combined']:<12.4f} {scores['perplexity']:<12.4f} "
                  f"{scores['coherence']:<10.4f}")
        
        # 显示最佳候选词
        if result['ranked_candidates']:
            best = result['ranked_candidates'][0]
            print("\n" + "=" * 70)
            print(f"✨ 最佳候选（最流畅）: {best['text']}")
            print(f"   综合分数: {best['scores']['combined']:.4f}")
            print(f"   困惑度: {best['scores']['perplexity']:.4f} (越低越好)")
            print(f"   连贯性: {best['scores']['coherence']:.4f}")
            print("=" * 70)
    
    def finalize(self):
        """清理资源"""
        if self.session_id:
            self.destroy_session()
        if self.rime:
            self.rime.finalize()
        print("\n资源已清理")


def demo_basic_usage():
    """基本使用示例（单次输入，不分段）"""
    print("\n" + "=" * 70)
    print("基本使用示例（单次输入）")
    print("=" * 70)
    
    # 初始化
    input_method = RimeBertInputMethod(
        bert_model_name='bert-base-chinese',
        use_mlm_model=True
    )
    
    try:
        # 创建会话
        input_method.create_session()
        
        # 测试用例
        test_cases = [
            "gegeguojiayougegeguojiadeguoge",  # 各个国家有各个国家的国歌
            # "tushuguanlidecangshu",  # 图书馆里的藏书
            # "congmingdeshurufa",     # 聪明的输入法
            
        ]
        
        for pinyin in test_cases:
            result = input_method.input_pinyin(pinyin)
            if result:
                input_method.display_results(result, top_n=5)
            print("\n")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input_method.finalize()


def demo_sentence_input():
    """整句输入示例（自动分段）"""
    print("\n" + "=" * 70)
    print("整句输入示例（自动分段 + BERT 评估）")
    print("=" * 70)
    
    # 初始化
    input_method = RimeBertInputMethod(
        bert_model_name='bert-base-chinese',
        use_mlm_model=True
    )
    
    try:
        # 创建会话
        input_method.create_session()
        
        # 测试用例（整句拼音，无空格）
        test_cases = [
            "gegeguojiayougegeguojiadeguoge",  # 各个国家有各个国家的国歌
            # "congmingdeshurufa",           # 聪明的输入法
            # "tushuguanlidecangshu",        # 图书馆里的藏书
            
        ]
        
        for pinyin in test_cases:
            result = input_method.input_sentence_pinyin(
                pinyin,
                max_segment_candidates=5,  # 每个分段最多5个候选词
                max_combinations=50        # 最多评估50个组合
            )
            if result:
                input_method.display_sentence_results(result, top_n=10)
            print("\n")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input_method.finalize()


def demo_interactive():
    """交互式示例"""
    print("\n" + "=" * 70)
    print("交互式输入示例")
    print("=" * 70)
    
    # 初始化
    input_method = RimeBertInputMethod(
        bert_model_name='bert-base-chinese',
        use_mlm_model=True
    )
    
    try:
        # 创建会话
        input_method.create_session()
        
        print("\n提示：")
        print("  - 输入拼音（全小写，无空格）")
        print("  - 输入 'quit' 或 'exit' 退出")
        print("  - 输入 'help' 查看帮助\n")
        
        while True:
            pinyin = input("请输入拼音: ").strip()
            
            if not pinyin:
                continue
            
            if pinyin.lower() in ['quit', 'exit', 'q']:
                print("退出...")
                break
            
            if pinyin.lower() == 'help':
                print("\n帮助：")
                print("  - 输入全小写拼音，例如：tushuguanlidecangshu")
                print("  - 系统会使用 BERT 模型对候选词进行流畅度评分")
                print("  - 显示排序后的候选词和评分信息\n")
                continue
            
            result = input_method.input_pinyin(pinyin)
            if result:
                input_method.display_results(result, top_n=10)
            print()
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input_method.finalize()


def demo_comparison():
    """对比示例：显示原始排序 vs BERT 排序"""
    print("\n" + "=" * 70)
    print("对比示例：原始排序 vs BERT 排序")
    print("=" * 70)
    
    # 初始化
    input_method = RimeBertInputMethod(
        bert_model_name='bert-base-chinese',
        use_mlm_model=True
    )
    
    try:
        # 创建会话
        input_method.create_session()
        
        # 测试用例
        pinyin = "tushuguanlidecangshu"
        result = input_method.input_pinyin(pinyin)
        
        if result:
            print("\n" + "=" * 70)
            print("原始排序（Rime 默认排序）")
            print("=" * 70)
            for i, cand in enumerate(result['original_candidates'][:10], 1):
                text = cand.get('text', '')
                comment = cand.get('comment', '')
                print(f"{i}. {text}" + (f" ({comment})" if comment else ""))
            
            print("\n" + "=" * 70)
            print("BERT 排序（按流畅度）")
            print("=" * 70)
            input_method.display_results(result, top_n=10)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input_method.finalize()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Rime + BERT 整句输入法演示')
    parser.add_argument('--mode', type=str,
                       choices=['basic', 'sentence', 'interactive', 'comparison'],
                       default='sentence',
                       help='运行模式: basic (基本示例), sentence (整句输入), interactive (交互式), comparison (对比示例)')
    parser.add_argument('--model', type=str,
                       default='bert-base-chinese',
                       help='BERT 模型名称（默认: bert-base-chinese）')
    
    args = parser.parse_args()
    
    if args.mode == 'basic':
        demo_basic_usage()
    elif args.mode == 'sentence':
        demo_sentence_input()
    elif args.mode == 'interactive':
        demo_interactive()
    elif args.mode == 'comparison':
        demo_comparison()

