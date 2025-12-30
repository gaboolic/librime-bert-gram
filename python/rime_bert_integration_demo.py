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
    
    def __init__(self, rime_dll_path=None, bert_model_name='bert-base-chinese', use_mlm_model=True, schema_name='luna_pinyin', device=None):
        """
        初始化输入法
        
        Args:
            rime_dll_path: rime.dll 的路径，如果为 None 会自动查找
            bert_model_name: BERT 模型名称
            use_mlm_model: 是否使用 MLM 模型
            schema_name: 输入方案名称（默认：luna_pinyin，可传入 rime_frost 等）
            device: 设备（'cuda', 'cpu' 或 None，None 时自动检测 GPU）
        """
        print("=" * 70)
        print("初始化 Rime + BERT 输入法")
        print("=" * 70)
        
        # 初始化 Rime
        print("\n[1/2] 初始化 Rime...")
        
        # 设置 DLL 和数据文件路径（在创建 RimeDllWrapper 之前）
        dll_dir = r"D:\Program Files\Rime\weasel-0.16.3"
        data_dir = r"D:\vscode\rime-frost"  # 使用指定的数据目录
        
        print(f"   DLL 目录: {dll_dir}")
        print(f"   数据目录: {data_dir}")
        
        # 查找 rime.dll（优先使用指定路径）
        if rime_dll_path:
            # 如果用户提供了 DLL 路径，直接使用
            dll_path = rime_dll_path
        else:
            # 否则尝试在 dll_dir 中查找
            dll_path = os.path.join(dll_dir, "rime.dll")
            if not os.path.exists(dll_path):
                # 如果指定路径不存在，设置为 None 让 RimeDllWrapper 自动查找
                dll_path = None
        
        # 创建 RimeDllWrapper 实例
        if dll_path and os.path.exists(dll_path):
            # 使用指定的 DLL 路径
            self.rime = RimeDllWrapper(dll_path=dll_path)
            print(f"   ✓ 找到 rime.dll: {self.rime.dll_path}")
        else:
            # 回退到自动查找（如果用户提供了路径但不存在，也使用自动查找）
            self.rime = RimeDllWrapper(dll_path=None)
            print(f"   ✓ 找到 rime.dll: {self.rime.dll_path}")
        
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
            use_mlm_model=use_mlm_model,
            device=device
        )
        print("   ✓ BERT 评分器初始化完成")
        
        # 保存默认输入方案
        self.default_schema_name = schema_name
        
        self.session_id = None
        print("\n" + "=" * 70)
        print("初始化完成！")
        print("=" * 70)
    
    def create_session(self, schema_name=None):
        """
        创建输入会话
        
        Args:
            schema_name: 输入方案名称（如果为 None，使用初始化时设置的默认方案）
        """
        if schema_name is None:
            schema_name = self.default_schema_name
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
        schema_name = self.rime.get_current_schema(self.session_id) or self.default_schema_name
        
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
        schema_name = self.rime.get_current_schema(self.session_id) or self.default_schema_name
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
                             max_combinations: int = 30) -> Optional[Dict]:
        """
        整句输入：参考 MakeSentence 逻辑，构建词图，生成所有可能的路径，用 BERT 评估
        
        参考 librime 的 MakeSentence 逻辑：
        1. 构建词图（WordGraph）：从每个位置开始，查找所有可能的词（不同长度）
        2. 生成所有可能的路径组合（使用评分筛选，减少无效路径）
        3. 使用 BERT 评估所有路径，找到最合理的整句
        
        优化点：
        - 词图构建时根据分段长度动态调整候选词数量（单字2个，双字词5个，3音节以上5个）
        - 路径生成时使用Rime排序评分，优先保留高质量路径
        - 每个位置最多保留50条路径（可配置）
        
        Args:
            pinyin_text: 整句拼音（全小写，无空格，例如："gegeguojiayougegeguojiadeguoge"）
            max_segment_candidates: 每个分段最多考虑的候选词数量（根据分段长度动态调整：单字2个，双字词5个，3音节以上5个）
            max_combinations: 最多评估的组合数量（避免组合爆炸，默认30）
        
        Returns:
            包含最佳完整句子和评分信息的字典
        """
        if not self.clear_input():
            return None
        
        print(f"\n输入整句拼音: {pinyin_text}")
        print(f"参考 MakeSentence 逻辑：构建词图 -> 生成路径 -> BERT 评估")
        
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
        
        # 4. 获取 Rime 直接返回的完整候选词（作为参考，但主要使用词图方法）
        candidates = context.get('candidates', [])
        rime_full_candidates = [cand.get('text', '') for cand in candidates if cand.get('text', '')]
        
        if rime_full_candidates:
            print(f"   ✓ Rime 返回了 {len(rime_full_candidates)} 个完整候选词（作为参考）")
            print(f"   前5个候选词: {', '.join(rime_full_candidates[:5])}")
        
        # 5. 构建词图（WordGraph）：参考 MakeSentence 的逻辑
        # WordGraph 结构：{start_pos: {end_pos: [候选词列表]}}
        print("\n构建词图（WordGraph）...")
        
        # 解析音节切分（按空格分割）
        syllables = input_text.strip().split()
        if not syllables:
            print("   ✗ 无法解析音节")
            return None
        
        print(f"   音节数量: {len(syllables)}")
        print(f"   音节列表: {syllables}")
        
        # 构建词图：从每个位置开始，尝试不同长度的分段
        # 类似 MakeSentence 中的逻辑：遍历所有可能的起始位置和长度
        word_graph = {}  # {start_pos: {end_pos: [(候选词, 排序)]}}
        total_syllable_length = len(''.join(syllables))  # 总拼音长度（无空格）
        
        # 优化：根据分段长度动态调整候选词数量
        # 对于较长的分段（2个音节以上），保留更多候选词，避免遗漏重要词汇
        # 对于单字，只取前1-2个即可
        
        # 从每个可能的起始位置开始查找
        for start_pos in range(len(syllables)):
            word_graph[start_pos] = {}
            
            # 优化：优先尝试更常见的词长度（2-3个音节最常见）
            # 常见长度：2(双字词), 3(三字词), 4(四字词), 1(单字), 5(五字词)
            # 优先顺序：2 > 3 > 4 > 1 > 5
            # 注意：优先尝试更长的词，避免过早使用单字，这样可以包含更多完整词汇（如"倉鼠"）
            lengths_to_try = [2, 3, 4, 1, 5]
            
            for length in lengths_to_try:
                if start_pos + length > len(syllables):
                    continue
                
                # 获取这个分段的拼音
                segment_syllables = syllables[start_pos:start_pos+length]
                segment_pinyin = ''.join(segment_syllables)  # 无空格连接
                end_pos = start_pos + length
                
                # 根据分段长度动态调整候选词数量
                # 单字：只取前1-2个（减少组合爆炸）
                # 2个音节以上：取更多候选词（避免遗漏重要词汇，如"倉鼠"）
                if length == 1:
                    max_candidates_for_segment = min(2, max_segment_candidates)
                elif length == 2:
                    # 双字词：取前5个，因为双字词很重要且数量相对较少
                    max_candidates_for_segment = min(5, max_segment_candidates)
                else:
                    # 3个音节以上：取更多候选词
                    max_candidates_for_segment = min(5, max_segment_candidates)
                
                # 获取这个分段的候选词
                candidates = self.get_candidates_for_pinyin(segment_pinyin, max_candidates_for_segment)
                if candidates:
                    # 存储候选词及其在Rime中的排序（用于后续路径评分）
                    word_graph[start_pos][end_pos] = [(cand, i) for i, cand in enumerate(candidates)]
                    # print(f"   位置 {start_pos}->{end_pos} ({segment_pinyin}, 长度{length}): {len(candidates)} 个候选词")
        
        # 打印词图信息（包含候选词详情，用于调试）
        print(f"\n词图构建完成:")
        total_edges = sum(len(edges) for edges in word_graph.values())
        print(f"   顶点数: {len(word_graph)}")
        print(f"   边数: {total_edges}")
        # 打印关键位置的候选词（用于调试）
        # print(f"\n关键位置的候选词详情:")
        # for start_pos in sorted(word_graph.keys()):
        #     for end_pos in sorted(word_graph[start_pos].keys()):
        #         candidates = word_graph[start_pos][end_pos]
        #         candidate_texts = [f"{word}(rank:{rank})" for word, rank in candidates]
        #         pinyin = ''.join(syllables[start_pos:end_pos])
        #         print(f"   位置 {start_pos}->{end_pos} ({pinyin}): {', '.join(candidate_texts)}")
        
        # 6. 生成所有可能的路径（从位置0到位置len(syllables)）
        # 使用动态规划 + 限制每个位置的路径数量（类似 Beam Search）
        print("\n生成所有可能的路径（使用 Beam Search 限制）...")
        
        # 使用动态规划：dp[pos] = 到达位置 pos 的所有路径（限制数量）
        # 每个路径是 [(start_pos, end_pos, word, rank), ...] 的列表
        # rank 是候选词在Rime中的排序（用于评分）
        dp = {}  # {pos: [路径列表]}
        # 优化：减少每个位置保留的路径数，使用评分来筛选高质量路径
        max_paths_per_pos = min(50, max_combinations)  # 每个位置最多保留的路径数（减少组合爆炸）
        
        # 初始化：位置0的路径为空路径
        # 路径格式：[(start_pos, end_pos, word, rank), ...]
        dp[0] = [[]]
        
        # 按位置顺序处理（从小到大），使用多轮迭代确保所有路径都被扩展
        target_pos = len(syllables)
        max_iterations = len(syllables) * 2  # 最多迭代次数，避免无限循环
        
        def calculate_path_score(path):
            """
            计算路径的评分（用于筛选高质量路径）
            评分规则：
            1. 路径中所有候选词的Rime排序之和（越小越好，rank=0表示最优）
            2. 路径长度（边数，越少越好，因为使用更长的词更符合中文习惯）
            3. 优先使用完整词汇（如"倉鼠"）而不是单字组合（如"倉"+"書"）
            """
            if not path:
                return float('inf')
            # 计算所有候选词的排序之和（rank越小越好）
            total_rank = sum(rank for _, _, _, rank in path)
            # 路径长度（边数，越少越好，因为使用更长的词更符合中文习惯）
            # 例如："糧倉裏的倉鼠"（3条边）比"糧倉裏的倉書"（4条边）更好
            path_length = len(path)
            # 综合评分：排序和越小越好，路径长度越小越好
            # 路径长度权重设为0.5，让使用更长词的路径评分更好
            return total_rank + path_length * 0.5
        
        for iteration in range(max_iterations):
            changed = False
            # 按位置顺序处理
            positions_to_process = sorted([pos for pos in dp.keys() if pos < target_pos])
            
            for pos in positions_to_process:
                if pos not in word_graph:
                    continue
                
                # 从当前位置出发的所有路径
                current_paths = dp[pos]
                if not current_paths:
                    continue
                
                # 遍历所有可能的边
                for end_pos, candidate_list in word_graph[pos].items():
                    if end_pos > target_pos:
                        continue  # 跳过超过终点的边
                    
                    if end_pos not in dp:
                        dp[end_pos] = []
                    
                    # 为每条当前路径，尝试所有候选词
                    # 路径格式：[(start_pos, end_pos, word, rank), ...]
                    new_paths = []
                    for path in current_paths:
                        for candidate, rank in candidate_list:
                            new_path = path + [(pos, end_pos, candidate, rank)]
                            # 检查是否已存在相同路径（避免重复）
                            path_str = ''.join(word for _, _, word, _ in new_path)
                            existing_paths_str = [''.join(word for _, _, word, _ in p) for p in dp[end_pos]]
                            if path_str not in existing_paths_str:
                                new_paths.append(new_path)
                    
                    if new_paths:
                        # 将新路径添加到目标位置
                        dp[end_pos].extend(new_paths)
                        changed = True
                        
                        # 限制目标位置的路径数量（保留前 N 条）
                        # 优化：使用评分来筛选，而不是简单按长度
                        if len(dp[end_pos]) > max_paths_per_pos:
                            # 按路径评分排序，优先保留评分更好的路径（评分越小越好）
                            dp[end_pos].sort(key=calculate_path_score)
                            dp[end_pos] = dp[end_pos][:max_paths_per_pos]
            
            # 如果没有变化，说明已经处理完所有可能的路径
            if not changed:
                break
        
        # 只获取到达终点的完整路径
        if target_pos in dp and dp[target_pos]:
            all_paths = dp[target_pos]
            # 对完整路径按评分排序
            all_paths.sort(key=calculate_path_score)
            print(f"   生成 {len(all_paths)} 条完整路径（到达终点位置 {target_pos}）")
            # 调试：只显示前10条路径（减少输出）
            if len(all_paths) > 0:
                print(f"   前30条路径详情:")
                for i, path in enumerate(all_paths[:30], 1):
                    path_str = ' -> '.join([f"{start}->{end}({word})" for start, end, word, _ in path])
                    sentence = ''.join(word for _, _, word, _ in path)
                    score = calculate_path_score(path)
                    print(f"      路径{i} (评分:{score:.2f}): {path_str} -> 句子: {sentence}")
        else:
            print(f"   ⚠ 无法到达终点位置 {target_pos}")
            # 显示到达的位置信息（用于调试）
            reached_positions = sorted([pos for pos in dp.keys() if pos <= target_pos])
            if reached_positions:
                print(f"   到达的位置: {reached_positions}")
                print(f"   最接近终点的位置: {max(reached_positions)}")
                # 显示词图中是否有到达终点的边
                has_target_edge = any(target_pos in edges for edges in word_graph.values())
                print(f"   词图中是否有到达终点的边: {has_target_edge}")
            all_paths = []
            print(f"   ✗ 无法生成完整路径，跳过不完整的路径")
        
        # 限制路径数量，避免组合爆炸
        if len(all_paths) > max_combinations:
            print(f"   路径过多，限制为前 {max_combinations} 条")
            all_paths = all_paths[:max_combinations]
        
        # 将路径转换为完整句子
        # 注意：路径已经通过动态规划确保到达终点，所以直接转换即可
        all_combinations = []
        seen_sentences = set()  # 用于去重
        target_pos = len(syllables)
        
        for path in all_paths:
            # 路径格式：[(start_pos, end_pos, word, rank), ...]
            # 验证路径是否真的覆盖了从0到target_pos的所有位置
            current_pos = 0
            is_valid = True
            
            for start_pos, end_pos, word, rank in path:
                # 检查路径是否连续
                if start_pos != current_pos:
                    is_valid = False
                    break
                # 更新当前位置
                current_pos = end_pos
            
            # 路径验证通过：检查是否到达了终点
            if is_valid and current_pos == target_pos:
                sentence = ''.join(word for _, _, word, _ in path)
                # 验证：句子长度应该合理
                # 每个音节平均对应1-2个汉字，但也要考虑特殊情况（如"的"、"了"等助词）
                # 设置一个合理的下限：至少70%的音节数，但不少于音节数-2
                min_length = max(int(len(syllables) * 0.7), len(syllables) - 2)
                
                if len(sentence) >= min_length and sentence not in seen_sentences:
                    seen_sentences.add(sentence)
                    all_combinations.append(sentence)
                # 如果句子太短，记录一下（用于调试）
                elif len(sentence) < min_length:
                    pass  # 句子太短，跳过
            # 如果路径无效，跳过（不添加到结果中）
        print(f"   ✓ 从词图生成 {len(all_combinations)} 个完整句子组合")
        
        # 6.5. 合并 Rime 的完整候选词（去重 + 过滤不完整的）
        if rime_full_candidates:
            # 过滤：只保留真正完整的句子
            # 完整句子的标准：长度应该至少覆盖大部分音节
            # 每个音节平均对应1-2个汉字，所以完整句子长度应该 >= 音节数的80%
            min_length_ratio = 0.8  # 至少覆盖80%的音节
            min_length = max(int(len(syllables) * min_length_ratio), len(syllables) - 2)  # 至少80%或音节数-2
            
            # 过滤掉太短的候选词（可能是部分匹配）
            filtered_rime_candidates = [
                c for c in rime_full_candidates 
                if len(c) >= min_length
            ]
            
            if filtered_rime_candidates:
                # 将 Rime 的候选词加入，但去重
                graph_set = set(all_combinations)
                new_from_rime = [c for c in filtered_rime_candidates if c not in graph_set]
                if new_from_rime:
                    print(f"   从 Rime 完整候选词中过滤后新增 {len(new_from_rime)} 个（原始 {len(rime_full_candidates)} 个，过滤掉 {len(rime_full_candidates) - len(filtered_rime_candidates)} 个不完整的）")
                    all_combinations.extend(new_from_rime[:max_combinations // 2])  # 最多加入一半
                else:
                    print(f"   Rime 候选词已全部包含在词图生成的路径中")
            else:
                print(f"   Rime 候选词全部被过滤（太短，不完整）")
        
        # 限制总数量
        if len(all_combinations) > max_combinations:
            print(f"   组合过多，限制为前 {max_combinations} 个")
            all_combinations = all_combinations[:max_combinations]
        
        print(f"   ✓ 总共 {len(all_combinations)} 个完整句子组合（词图生成 + Rime 候选词）")
        
        # 输出前几个组合（用于调试，减少输出）
        # print(f"\n生成的完整句子组合（前10个）:")
        # for i, combo in enumerate(all_combinations, 1):
        #     print(f"   {i}. {combo}")

        
        # 检查是否包含"图书馆里的藏书"（忽略繁简）
        # target_sentences = [ "糧倉裏的倉鼠"]
        # found_target = [s for s in all_combinations if s in target_sentences]
        # if found_target:
        #     print(f"\n   ✓ 找到目标句子: {found_target}")
        # else:
        #     print(f"\n   ⚠ 未找到目标句子'图书馆里的藏书'（已检查 {len(all_combinations)} 个组合）")
        #     # 检查是否有类似的句子
        #     similar = [s for s in all_combinations if "圖書館" in s and "藏書" in s]
        #     if similar:
        #         print(f"   类似的句子: {similar[:5]}")
        
        # 7. 使用 BERT 评估所有组合
        print(f"\n使用 BERT 模型评估 {len(all_combinations)} 个完整句子...")
        start_time = time.time()
        
        ranked_results = self.scorer.compare_sentences(all_combinations, method='combined')
        
        elapsed_time = (time.time() - start_time) * 1000
        print(f"   ✓ 评估完成（耗时: {elapsed_time:.2f} ms）")
        
        # 8. 构建结果
        # 将词图转换为可显示的格式
        word_graph_display = {}
        for start_pos, edges in word_graph.items():
            word_graph_display[start_pos] = {}
            for end_pos, candidate_list in edges.items():
                # candidate_list 是 [(word, rank), ...] 格式
                candidates = [word for word, _ in candidate_list]
                word_graph_display[start_pos][end_pos] = {
                    'pinyin': ''.join(syllables[start_pos:end_pos]),
                    'candidates': candidates[:3]  # 只保存前3个用于显示
                }
        
        result = {
            'input_pinyin': pinyin_text,
            'rime_input': input_text,  # Rime 切分后的拼音
            'syllables': syllables,  # 音节列表
            'word_graph': word_graph_display,  # 词图信息（用于显示）
            'rime_full_candidates_count': len(rime_full_candidates) if rime_full_candidates else 0,
            'total_paths': len(all_paths),
            'total_combinations': len(all_combinations),
            'ranked_sentences': [],
            'scoring_time_ms': elapsed_time,
            'method': 'word_graph_paths'  # 标记使用的方法
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
        print(f"方法: {result.get('method', 'unknown')}")
        if 'total_paths' in result:
            print(f"词图生成路径: {result['total_paths']} 条")
        if 'rime_full_candidates_count' in result and result['rime_full_candidates_count'] > 0:
            print(f"Rime 完整候选词: {result['rime_full_candidates_count']} 个（已合并）")
        print(f"总评估组合: {result['total_combinations']} 个")
        print(f"评估耗时: {result['scoring_time_ms']:.2f} ms")
        
        # 显示词图信息（如果存在）
        # if 'word_graph' in result and result['word_graph']:
        #     print(f"\n词图信息:")
        #     for start_pos, edges in sorted(result['word_graph'].items()):
        #         for end_pos, info in sorted(edges.items()):
        #             pinyin = info.get('pinyin', '')
        #             candidates = info.get('candidates', [])
        #             candidates_str = ', '.join(candidates) if candidates else '无'
        #             print(f"   位置 {start_pos}->{end_pos} ({pinyin}): {candidates_str}")
        
        # 显示分段信息（如果存在）
        if 'segments' in result and result['segments']:
            print(f"\n分段信息:")
            for i, seg in enumerate(result['segments']):
                print(f"  分段 {i+1}: {seg.get('pinyin', '')}")
        
        # 显示音节信息
        if 'syllables' in result and result['syllables']:
            print(f"\n音节列表: {' '.join(result['syllables'])}")
        
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



def demo_sentence_input(bert_model_name='bert-base-chinese', device=None):
    """整句输入示例（自动分段）"""
    print("\n" + "=" * 70)
    print("整句输入示例（自动分段 + BERT 评估）")
    print("=" * 70)
    
    # 初始化（使用 rime_frost 输入方案）
    # 如果需要使用默认的 luna_pinyin，可以不传 schema_name 参数
    input_method = RimeBertInputMethod(
        bert_model_name=bert_model_name,
        use_mlm_model=True,
        schema_name='rime_frost',  # 使用 rime_frost 方案，默认是 luna_pinyin
        device=device
    )
    
    try:
        # 创建会话
        input_method.create_session()
        
        # 测试用例（整句拼音，无空格）
        test_cases = [
            # "gegeguojiayougegeguojiadeguoge",  # 各个国家有各个国家的国歌
            # "congmingdeshurufa",           # 聪明的输入法
            # "tushuguanlidecangshu",        # 图书馆里的藏书
            "liangcanglidecangshu",        # 粮仓里的仓鼠
            # "haerbinzhidongbuzaijimo",
            # "muqianhexifuquanzhijujiazuokuajingxiaoshuochuhai",
            # "youshijianyidingshiyongyixia"
        ]
        
        for pinyin in test_cases:
            result = input_method.input_sentence_pinyin(
                pinyin,
                max_segment_candidates=5,  # 每个分段最多5个候选词（根据分段长度动态调整）
                max_combinations=30        # 最多评估30个组合（减少无效路径）
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


def demo_interactive(bert_model_name='bert-base-chinese', device=None):
    """交互式示例"""
    print("\n" + "=" * 70)
    print("交互式输入示例")
    print("=" * 70)
    
    # 初始化
    input_method = RimeBertInputMethod(
        bert_model_name=bert_model_name,
        use_mlm_model=True,
        device=device
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


def demo_comparison(bert_model_name='bert-base-chinese', device=None):
    """对比示例：显示原始排序 vs BERT 排序"""
    print("\n" + "=" * 70)
    print("对比示例：原始排序 vs BERT 排序")
    print("=" * 70)
    
    # 初始化
    input_method = RimeBertInputMethod(
        bert_model_name=bert_model_name,
        use_mlm_model=True,
        device=device
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
    
    # bert_model_name='bert-base-chinese'   基础BERT模型
    # 或 bert_model_name='ckiplab/bert-tiny-chinese', 蒸馏的小BERT模型（速度快但是效果不好）

    parser = argparse.ArgumentParser(description='Rime + BERT 整句输入法演示')
    parser.add_argument('--mode', type=str,
                       choices=[ 'sentence', 'interactive', 'comparison'],
                       default='sentence',
                       help='运行模式:  sentence (整句输入), interactive (交互式), comparison (对比示例)')
    parser.add_argument('--model', type=str,
                       default='bert-base-chinese',
                       help='BERT 模型名称（默认: bert-base-chinese）')
    parser.add_argument('--device', type=str,
                       choices=['cuda', 'cpu', 'auto'],
                       default='auto',
                       help='计算设备: cuda (GPU), cpu (CPU), auto (自动检测，默认)')
    
    args = parser.parse_args()
    print("args: "+str(args))
    
    # 处理设备参数
    device = None if args.device == 'auto' else args.device
    
    if args.mode == 'sentence':
        demo_sentence_input(bert_model_name=args.model, device=device)
    elif args.mode == 'interactive':
        demo_interactive(bert_model_name=args.model, device=device)
    elif args.mode == 'comparison':
        demo_comparison(bert_model_name=args.model, device=device)

