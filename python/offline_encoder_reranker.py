"""
使用 encoder-only ONNX 模型做离线候选重排实验。

说明：
- 这是一个“验证区分度”的小脚本，不是最终可上线的排序器。
- 由于当前只有 encoder，没有专门训练过的 reranker head，这里使用一个简单启发式：
  1. 把输入构造成 [CLS] context [SEP] candidate [SEP]
  2. 分别对 context token 和 candidate token 做 mean pooling
  3. 用 cosine(context_emb, candidate_emb) 作为相关性分数
  4. 再和整句 [CLS] 向量范数做一个很轻的混合，得到最终分数
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


DEFAULT_MODEL_DIR = Path(__file__).parent / "onnx_models" / "ernie-3.0-nano-zh"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "nghuyong_ernie-3.0-nano-zh_encoder.onnx"


@dataclass
class CandidateScore:
    candidate: str
    score: float
    cosine_score: float
    cls_norm_score: float
    context_tokens: int
    candidate_tokens: int


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return vec
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = l2_normalize(a)
    b_norm = l2_normalize(b)
    return float(np.dot(a_norm, b_norm))


class OfflineEncoderReranker:
    def __init__(self, model_path: Path, tokenizer_path: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [item.name for item in self.session.get_inputs()]

    def _encode_pair(self, context: str, candidate: str):
        encoded = self.tokenizer(
            context,
            candidate,
            return_tensors="np",
            return_token_type_ids=True,
            truncation=True,
            max_length=128,
        )
        feed = {
            key: value.astype(np.int64)
            for key, value in encoded.items()
            if key in self.input_names
        }
        outputs = self.session.run(None, feed)
        last_hidden_state = outputs[0][0]  # [seq_len, hidden]
        pooler_output = outputs[1][0]      # [hidden]
        input_ids = encoded["input_ids"][0]
        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids[0]
        attention_mask = encoded["attention_mask"][0]
        return input_ids, token_type_ids, attention_mask, last_hidden_state, pooler_output

    def score_candidate(self, context: str, candidate: str) -> CandidateScore:
        input_ids, token_type_ids, attention_mask, hidden, pooler = self._encode_pair(context, candidate)

        valid_len = int(attention_mask.sum())
        hidden = hidden[:valid_len]
        input_ids = input_ids[:valid_len]

        if token_type_ids is None:
            # 如果 tokenizer 没返回 token_type_ids，就退化成用 [SEP] 位置切分。
            sep_positions = np.where(input_ids == self.tokenizer.sep_token_id)[0]
            if len(sep_positions) >= 2:
                first_sep = int(sep_positions[0])
                second_sep = int(sep_positions[1])
                context_slice = slice(1, first_sep)
                candidate_slice = slice(first_sep + 1, second_sep)
            else:
                midpoint = max(1, valid_len // 2)
                context_slice = slice(1, midpoint)
                candidate_slice = slice(midpoint, valid_len - 1)
        else:
            token_type_ids = token_type_ids[:valid_len]
            context_positions = np.where(token_type_ids == 0)[0]
            candidate_positions = np.where(token_type_ids == 1)[0]
            context_positions = context_positions[input_ids[context_positions] != self.tokenizer.cls_token_id]
            context_positions = context_positions[input_ids[context_positions] != self.tokenizer.sep_token_id]
            candidate_positions = candidate_positions[input_ids[candidate_positions] != self.tokenizer.sep_token_id]

            if len(context_positions) == 0:
                context_slice = slice(1, 2)
            else:
                context_slice = slice(int(context_positions[0]), int(context_positions[-1]) + 1)

            if len(candidate_positions) == 0:
                candidate_slice = slice(max(1, valid_len - 2), max(2, valid_len - 1))
            else:
                candidate_slice = slice(int(candidate_positions[0]), int(candidate_positions[-1]) + 1)

        context_hidden = hidden[context_slice]
        candidate_hidden = hidden[candidate_slice]

        context_emb = context_hidden.mean(axis=0)
        candidate_emb = candidate_hidden.mean(axis=0)
        cosine_score = cosine_similarity(context_emb, candidate_emb)

        # [CLS] / pooler 的范数只能当很弱的句对“强度”信号，权重故意放小。
        cls_norm_score = float(np.linalg.norm(pooler) / math.sqrt(pooler.shape[0]))
        final_score = cosine_score + 0.05 * cls_norm_score

        return CandidateScore(
            candidate=candidate,
            score=final_score,
            cosine_score=cosine_score,
            cls_norm_score=cls_norm_score,
            context_tokens=len(context_hidden),
            candidate_tokens=len(candidate_hidden),
        )

    def rank_candidates(self, context: str, candidates: Sequence[str]) -> List[CandidateScore]:
        results = [self.score_candidate(context, candidate) for candidate in candidates]
        return sorted(results, key=lambda item: item.score, reverse=True)


def print_results(title: str, context: str, ranked: Sequence[CandidateScore]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"context: {context!r}")
    print("ranking:")
    for index, item in enumerate(ranked, start=1):
        print(
            f"  {index}. {item.candidate}"
            f" | score={item.score:.6f}"
            f" | cosine={item.cosine_score:.6f}"
            f" | cls_norm={item.cls_norm_score:.6f}"
            f" | ctx_tok={item.context_tokens}"
            f" | cand_tok={item.candidate_tokens}"
        )


def run_builtin_cases(reranker: OfflineEncoderReranker) -> None:
    cases = [
        {
            "title": "Case 1: 国歌短语",
            "context": "各个国家有各个国家的",
            "candidates": ["国歌", "德国个", "个", "歌曲"],
        },
        {
            "title": "Case 2: 天气短语",
            "context": "今天北京的天气",
            "candidates": ["不错", "葡萄", "很好", "很差"],
        },
        {
            "title": "Case 3: 常见输入法候选",
            "context": "这个问题需要",
            "candidates": ["解决", "介绍", "结果", "结构"],
        },
    ]

    for case in cases:
        ranked = reranker.rank_candidates(case["context"], case["candidates"])
        print_results(case["title"], case["context"], ranked)


def parse_args():
    parser = argparse.ArgumentParser(description="离线验证 encoder-only ONNX 候选排序区分度")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"ONNX encoder 模型路径（默认: {DEFAULT_MODEL_PATH}）",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"tokenizer 目录（默认: {DEFAULT_MODEL_DIR}）",
    )
    parser.add_argument("--context", type=str, default="", help="要测试的上下文")
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=[],
        help="候选词列表，例如 --candidates 国歌 德国个 歌曲",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    reranker = OfflineEncoderReranker(args.model, args.tokenizer)

    if args.context and args.candidates:
        ranked = reranker.rank_candidates(args.context, args.candidates)
        print_results("Custom Case", args.context, ranked)
    else:
        run_builtin_cases(reranker)


if __name__ == "__main__":
    main()
