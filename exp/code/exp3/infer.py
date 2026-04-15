from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import BertTokenizerFast

from Dataset import SquadExtractiveDataset, SquadGenerativeDataset
from QA1 import ExtractiveQA1
from QA2 import GenerativeQA2


def _log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}][Infer] {msg}")


def _build_qa1_from_config(cfg: Dict[str, Any]) -> ExtractiveQA1:
    return ExtractiveQA1(
        vocab_size=int(cfg["vocab_size"]),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_encoder_layers=int(cfg["num_encoder_layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg.get("dropout", 0.1)),
        max_len=int(cfg.get("max_len", 512)),
    )


def _build_qa2_from_config(cfg: Dict[str, Any]) -> GenerativeQA2:
    return GenerativeQA2(
        vocab_size=int(cfg["vocab_size"]),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_encoder_layers=int(cfg["num_encoder_layers"]),
        num_decoder_layers=int(cfg["num_decoder_layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg.get("dropout", 0.1)),
        max_len=int(cfg.get("max_len", 512)),
    )


class InferenceRunner:
    """加载最优 checkpoint，在验证集上随机抽样推理并写入日志。"""

    def __init__(
        self,
        checkpoint_dir: str | Path = "./exp/checkpoint",
        device: Optional[torch.device] = None,
        log_name: str = "infer.log",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_path = self.checkpoint_dir / log_name

    def _append_log(self, lines: List[str]) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().isoformat(timespec="seconds")
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n==== {stamp} ====\n")
            for line in lines:
                f.write(line + "\n")
        _log(f"推理日志已写入: {self.log_path}")

    def run_qa1_random(
        self,
        ckpt_path: str | Path,
        dev_json: str | Path,
        num_samples: int = 5,
        pool_size: int = 2000,
        max_length: int = 256,
        max_answer_len: int = 30,
    ) -> None:
        _log(f"QA1 推理开始 | ckpt={ckpt_path}, dev_json={dev_json}, num_samples={num_samples}, pool_size={pool_size}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        cfg = ckpt["config"]
        tokenizer = BertTokenizerFast.from_pretrained(cfg["tokenizer_name"])
        model = _build_qa1_from_config(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()

        ds = SquadExtractiveDataset(
            dev_json,
            tokenizer_name=cfg["tokenizer_name"],
            max_length=max_length,
            max_samples=pool_size,
        )
        if len(ds) == 0:
            self._append_log(["QA1: 验证样本为空，跳过。"])
            return
        _log(f"QA1 推理数据集加载完成 | 可采样样本={len(ds)}")
        idxs = random.sample(range(len(ds)), k=min(num_samples, len(ds)))
        lines: List[str] = []
        for i in idxs:
            batch = ds[i]
            input_ids = batch["input_ids"].unsqueeze(0).to(self.device)
            attn = batch["attention_mask"].unsqueeze(0).to(self.device)
            tt = batch.get("token_type_ids")
            if tt is not None:
                tt = tt.unsqueeze(0).to(self.device)
            with torch.no_grad():
                start_logits, end_logits, _ = model(input_ids, attn, tt, None, None)
            s_logits = start_logits[0].clone()
            e_logits = end_logits[0].clone()
            if tt is not None:
                # 只允许在 context(token_type_id=1) 中预测答案，避免把问题文本当答案
                context_mask = tt[0] == 1
                s_logits = s_logits.masked_fill(~context_mask, -1e4)
                e_logits = e_logits.masked_fill(~context_mask, -1e4)
            # 在 (s <= e, e-s+1 <= max_answer_len) 约束下选最佳 span
            best_score = -1e18
            s, e = 0, 0
            seq_len = s_logits.size(0)
            for si in range(seq_len):
                ei_max = min(seq_len - 1, si + max_answer_len - 1)
                e_slice = e_logits[si : ei_max + 1]
                val, rel_idx = torch.max(e_slice, dim=0)
                score = s_logits[si] + val
                if score.item() > best_score:
                    best_score = score.item()
                    s = si
                    e = si + int(rel_idx.item())
            pred_ids = input_ids[0, s : e + 1].tolist()
            pred = tokenizer.decode(pred_ids, skip_special_tokens=True)
            gold = batch["answer_text"]
            lines.append(f"[QA1] id={batch['id']}")
            lines.append(f"  gold: {gold}")
            lines.append(f"  pred: {pred}")
        self._append_log(lines)
        _log("QA1 推理结束")

    def run_qa2_random(
        self,
        ckpt_path: str | Path,
        dev_json: str | Path,
        num_samples: int = 5,
        pool_size: int = 2000,
        max_enc_length: int = 256,
        max_new_tokens: int = 48,
        min_new_tokens: int = 2,
        repetition_penalty: float = 1.1,
    ) -> None:
        _log(f"QA2 推理开始 | ckpt={ckpt_path}, dev_json={dev_json}, num_samples={num_samples}, pool_size={pool_size}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        cfg = ckpt["config"]
        tokenizer = BertTokenizerFast.from_pretrained(cfg["tokenizer_name"])
        model = _build_qa2_from_config(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)

        ds = SquadGenerativeDataset(
            dev_json,
            tokenizer_name=cfg["tokenizer_name"],
            max_enc_length=max_enc_length,
            max_dec_length=int(cfg.get("max_dec_length", 96)),
            max_samples=pool_size,
        )
        if len(ds) == 0:
            self._append_log(["QA2: 验证样本为空，跳过。"])
            return
        _log(f"QA2 推理数据集加载完成 | 可采样样本={len(ds)}")
        idxs = random.sample(range(len(ds)), k=min(num_samples, len(ds)))
        lines: List[str] = []
        pad_id = tokenizer.pad_token_id
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        for i in idxs:
            batch = ds[i]
            input_ids = batch["input_ids"].unsqueeze(0).to(self.device)
            attn = batch["attention_mask"].unsqueeze(0).to(self.device)
            gen = model.greedy_generate(
                input_ids,
                attn,
                max_new_tokens,
                pad_id,
                cls_id,
                sep_id,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
            )
            pred_ids = gen[0].tolist()
            if sep_id in pred_ids:
                cut = pred_ids.index(sep_id)
                pred_ids = pred_ids[: cut + 1]
            pred = tokenizer.decode(pred_ids, skip_special_tokens=True)
            gold = batch["answer_text"]
            lines.append(f"[QA2] id={batch['id']}")
            lines.append(f"  gold: {gold}")
            lines.append(f"  pred: {pred}")
        self._append_log(lines)
        _log("QA2 推理结束")
