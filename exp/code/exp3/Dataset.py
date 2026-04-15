from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


def _log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}][Dataset] {msg}")


def _read_squad(path: Path) -> List[Dict[str, Any]]:
    _log(f"开始读取 SQuAD 文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    rows: List[Dict[str, Any]] = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                qid = qa["id"]
                question = qa["question"]
                is_impossible = qa.get("is_impossible", False)
                answers = qa.get("answers", [])
                rows.append(
                    {
                        "id": qid,
                        "question": question,
                        "context": context,
                        "is_impossible": is_impossible,
                        "answers": answers,
                    }
                )
    _log(f"完成读取，共解析 QA 条目: {len(rows)}")
    return rows


def _char_span_to_token_span(
    offsets: List[Tuple[int, int]], answer_start: int, answer_end: int
) -> Optional[Tuple[int, int]]:
    start_index = None
    end_index = None
    for i, (cs, ce) in enumerate(offsets):
        if cs is None or ce is None:
            continue
        if start_index is None and cs <= answer_start < ce:
            start_index = i
        if cs < answer_end <= ce:
            end_index = i
            break
    if start_index is None or end_index is None:
        return None
    if end_index < start_index:
        return None
    return start_index, end_index


class SquadExtractiveDataset(Dataset):
    """抽取式：BERT 风格 [CLS] Q [SEP] C [SEP]，返回 start/end 词片下标。"""

    def __init__(
        self,
        json_path: str | Path,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 256,
        max_samples: Optional[int] = None,
        skip_impossible: bool = True,
    ) -> None:
        _log(
            f"构建抽取式数据集 | tokenizer={tokenizer_name}, max_length={max_length}, "
            f"max_samples={max_samples}, skip_impossible={skip_impossible}"
        )
        self.path = Path(json_path)
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.examples: List[Dict[str, Any]] = []
        skipped_impossible = 0
        skipped_no_answer = 0
        skipped_empty = 0
        skipped_unaligned = 0
        for row in _read_squad(self.path):
            if skip_impossible and row["is_impossible"]:
                skipped_impossible += 1
                continue
            if not row["answers"]:
                skipped_no_answer += 1
                continue
            ans = row["answers"][0]
            text = ans["text"]
            if not text.strip():
                skipped_empty += 1
                continue
            enc = self.tokenizer(
                row["question"],
                row["context"],
                truncation="only_second",
                max_length=max_length,
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            answer_end = ans["answer_start"] + len(text)
            span = _char_span_to_token_span(offsets, ans["answer_start"], answer_end)
            if span is None:
                skipped_unaligned += 1
                continue
            start, end = span
            item = {k: v[0] for k, v in enc.items()}
            item["start_positions"] = torch.tensor(start, dtype=torch.long)
            item["end_positions"] = torch.tensor(end, dtype=torch.long)
            item["id"] = row["id"]
            item["answer_text"] = text
            self.examples.append(item)
            if max_samples is not None and len(self.examples) >= max_samples:
                break
        _log(
            "抽取式数据集完成 | "
            f"可用样本={len(self.examples)}, "
            f"跳过 impossible={skipped_impossible}, 无答案={skipped_no_answer}, "
            f"空答案={skipped_empty}, 对齐失败={skipped_unaligned}"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


class SquadGenerativeDataset(Dataset):
    """生成式：编码器输入为 Q+C；解码器目标为答案 token 序列。"""

    def __init__(
        self,
        json_path: str | Path,
        tokenizer_name: str = "bert-base-uncased",
        max_enc_length: int = 256,
        max_dec_length: int = 96,
        max_samples: Optional[int] = None,
        skip_impossible: bool = True,
    ) -> None:
        _log(
            f"构建生成式数据集 | tokenizer={tokenizer_name}, max_enc_length={max_enc_length}, "
            f"max_dec_length={max_dec_length}, max_samples={max_samples}, skip_impossible={skip_impossible}"
        )
        self.path = Path(json_path)
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_enc_length = max_enc_length
        self.max_dec_length = max_dec_length
        self.examples: List[Dict[str, Any]] = []
        skipped_impossible = 0
        skipped_no_answer = 0
        skipped_empty = 0
        for row in _read_squad(self.path):
            if skip_impossible and row["is_impossible"]:
                skipped_impossible += 1
                continue
            if not row["answers"]:
                skipped_no_answer += 1
                continue
            text = row["answers"][0]["text"]
            if not text.strip():
                skipped_empty += 1
                continue
            self.examples.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "context": row["context"],
                    "answer_text": text,
                }
            )
            if max_samples is not None and len(self.examples) >= max_samples:
                break
        _log(
            "生成式数据集完成 | "
            f"可用样本={len(self.examples)}, "
            f"跳过 impossible={skipped_impossible}, 无答案={skipped_no_answer}, 空答案={skipped_empty}"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["question"],
            ex["context"],
            truncation="only_second",
            max_length=self.max_enc_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v[0] for k, v in enc.items()}

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id
        ans_ids = self.tokenizer.encode(ex["answer_text"], add_special_tokens=False)
        seq = [cls_id] + ans_ids + [sep_id]
        seq = seq[: self.max_dec_length]
        if len(seq) < self.max_dec_length:
            seq = seq + [pad_id] * (self.max_dec_length - len(seq))

        full = torch.tensor(seq, dtype=torch.long)
        dec_input_ids = full[:-1].clone()
        lab = full[1:].clone()
        lab = lab.masked_fill(lab == pad_id, -100)

        out = {**enc}
        out["decoder_input_ids"] = dec_input_ids
        out["decoder_labels"] = lab
        out["id"] = ex["id"]
        out["answer_text"] = ex["answer_text"]
        return out


def squad_extractive_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [k for k in batch[0].keys() if k not in ("id", "answer_text")]
    out: Dict[str, Any] = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["id"] = [b["id"] for b in batch]
    out["answer_text"] = [b["answer_text"] for b in batch]
    return out


def squad_generative_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [k for k in batch[0].keys() if k not in ("id", "answer_text")]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    out["id"] = [b["id"] for b in batch]
    out["answer_text"] = [b["answer_text"] for b in batch]
    return out
