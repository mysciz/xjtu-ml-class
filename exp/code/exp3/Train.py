from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import SquadExtractiveDataset, SquadGenerativeDataset, squad_extractive_collate, squad_generative_collate


def _build_optimizer(model: torch.nn.Module, name: str, lr: float) -> torch.optim.Optimizer:
    n = name.lower().strip()
    if n == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if n == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    if n == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    raise ValueError(f"不支持的优化方法: {name}，可选 adam / adamw / sgd")


def _log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}][Train] {msg}")


class QA1Trainer:
    """抽取式 QA1 训练器；仅保存验证集上最优 checkpoint。"""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: SquadExtractiveDataset,
        val_dataset: SquadExtractiveDataset,
        batch_size: int,
        optimizer_name: str,
        learning_rate: float,
        embedding_dim: int,
        num_epochs: int = 1,
        checkpoint_dir: str | Path = "./exp/checkpoint",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = _build_optimizer(self.model, optimizer_name, learning_rate)
        self.best_val = float("inf")
        self.best_path: Optional[Path] = None
        _log(
            f"初始化 QA1Trainer | device={self.device}, batch_size={self.batch_size}, "
            f"optimizer={self.optimizer_name}, lr={self.learning_rate}, embed_dim={self.embedding_dim}, "
            f"train_size={len(self.train_dataset)}, val_size={len(self.val_dataset)}"
        )

    def _dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=squad_extractive_collate,
            num_workers=0,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=squad_extractive_collate,
            num_workers=0,
        )
        _log(f"QA1 DataLoader 就绪 | train_steps={len(train_loader)}, val_steps={len(val_loader)}")
        return train_loader, val_loader

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total = 0.0
        n = 0
        start_correct = 0
        end_correct = 0
        span_correct = 0
        for batch in tqdm(loader, desc="QA1 Valid", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attn = batch["attention_mask"].to(self.device)
            token_type = batch.get("token_type_ids")
            if token_type is not None:
                token_type = token_type.to(self.device)
            start = batch["start_positions"].to(self.device)
            end = batch["end_positions"].to(self.device)
            _, _, loss = self.model(input_ids, attn, token_type, start, end)
            if loss is None:
                continue
            # 重新前向拿 logits，避免改动原返回结构
            start_logits, end_logits, _ = self.model(input_ids, attn, token_type, None, None)
            pred_s = torch.argmax(start_logits, dim=-1)
            pred_e = torch.argmax(end_logits, dim=-1)
            start_correct += (pred_s == start).sum().item()
            end_correct += (pred_e == end).sum().item()
            span_correct += ((pred_s == start) & (pred_e == end)).sum().item()
            bs = input_ids.size(0)
            total += loss.item() * bs
            n += bs
        denom = max(n, 1)
        return {
            "loss": total / denom,
            "start_acc": start_correct / denom,
            "end_acc": end_correct / denom,
            "span_acc": span_correct / denom,
        }

    def fit(self, tokenizer_name: str, extra_config: Optional[Dict[str, Any]] = None) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_loader, val_loader = self._dataloaders()
        cfg_base = {
            "task": "qa1",
            "tokenizer_name": tokenizer_name,
            "embedding_dim": self.embedding_dim,
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }
        if extra_config:
            cfg_base.update(extra_config)

        _log("QA1 开始训练")
        for epoch in range(self.num_epochs):
            self.model.train()
            _log(f"QA1 Epoch {epoch + 1}/{self.num_epochs} 开始")
            epoch_bar = tqdm(train_loader, desc=f"QA1 Train Epoch {epoch + 1}/{self.num_epochs}", leave=True)
            for batch in epoch_bar:
                input_ids = batch["input_ids"].to(self.device)
                attn = batch["attention_mask"].to(self.device)
                token_type = batch.get("token_type_ids")
                if token_type is not None:
                    token_type = token_type.to(self.device)
                start = batch["start_positions"].to(self.device)
                end = batch["end_positions"].to(self.device)
                _, _, loss = self.model(input_ids, attn, token_type, start, end)
                assert loss is not None
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                epoch_bar.set_postfix(loss=f"{loss.item():.4f}")

            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["loss"]
            _log(
                f"QA1 Epoch {epoch + 1}/{self.num_epochs} 结束 | "
                f"val_loss={val_loss:.4f}, start_acc={val_metrics['start_acc']:.4f}, "
                f"end_acc={val_metrics['end_acc']:.4f}, span_acc={val_metrics['span_acc']:.4f}, "
                f"best_loss={self.best_val:.4f}"
            )
            if val_loss < self.best_val:
                self.best_val = val_loss
                path = self.checkpoint_dir / "qa1_best.pt"
                payload = {
                    "model_state_dict": self.model.state_dict(),
                    "config": cfg_base,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                }
                torch.save(payload, path)
                self.best_path = path
                _log(f"QA1 保存最优模型: {path}")
        assert self.best_path is not None
        _log(f"QA1 训练完成 | best_ckpt={self.best_path}, best_val_loss={self.best_val:.4f}")
        return self.best_path


class QA2Trainer:
    """生成式 QA2 训练器；仅保存验证集上最优 checkpoint。"""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: SquadGenerativeDataset,
        val_dataset: SquadGenerativeDataset,
        batch_size: int,
        optimizer_name: str,
        learning_rate: float,
        embedding_dim: int,
        num_epochs: int = 1,
        checkpoint_dir: str | Path = "./exp/checkpoint",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = _build_optimizer(self.model, optimizer_name, learning_rate)
        self.best_val = float("inf")
        self.best_path: Optional[Path] = None
        _log(
            f"初始化 QA2Trainer | device={self.device}, batch_size={self.batch_size}, "
            f"optimizer={self.optimizer_name}, lr={self.learning_rate}, embed_dim={self.embedding_dim}, "
            f"train_size={len(self.train_dataset)}, val_size={len(self.val_dataset)}"
        )

    def _dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=squad_generative_collate,
            num_workers=0,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=squad_generative_collate,
            num_workers=0,
        )
        _log(f"QA2 DataLoader 就绪 | train_steps={len(train_loader)}, val_steps={len(val_loader)}")
        return train_loader, val_loader

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total = 0.0
        n = 0
        token_correct = 0
        token_total = 0
        for batch in tqdm(loader, desc="QA2 Valid", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attn = batch["attention_mask"].to(self.device)
            dec_in = batch["decoder_input_ids"].to(self.device)
            dec_lab = batch["decoder_labels"].to(self.device)
            logits, loss = self.model(input_ids, attn, dec_in, dec_lab)
            if loss is None:
                continue
            pred = torch.argmax(logits, dim=-1)
            valid = dec_lab != -100
            token_correct += ((pred == dec_lab) & valid).sum().item()
            token_total += valid.sum().item()
            bs = input_ids.size(0)
            total += loss.item() * bs
            n += bs
        loss_avg = total / max(n, 1)
        ppl = math.exp(loss_avg) if loss_avg < 20 else float("inf")
        return {
            "loss": loss_avg,
            "token_acc": token_correct / max(token_total, 1),
            "ppl": ppl,
        }

    def fit(self, tokenizer_name: str, extra_config: Optional[Dict[str, Any]] = None) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_loader, val_loader = self._dataloaders()
        cfg_base = {
            "task": "qa2",
            "tokenizer_name": tokenizer_name,
            "embedding_dim": self.embedding_dim,
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }
        if extra_config:
            cfg_base.update(extra_config)

        _log("QA2 开始训练")
        for epoch in range(self.num_epochs):
            self.model.train()
            _log(f"QA2 Epoch {epoch + 1}/{self.num_epochs} 开始")
            epoch_bar = tqdm(train_loader, desc=f"QA2 Train Epoch {epoch + 1}/{self.num_epochs}", leave=True)
            for batch in epoch_bar:
                input_ids = batch["input_ids"].to(self.device)
                attn = batch["attention_mask"].to(self.device)
                dec_in = batch["decoder_input_ids"].to(self.device)
                dec_lab = batch["decoder_labels"].to(self.device)
                _, loss = self.model(input_ids, attn, dec_in, dec_lab)
                assert loss is not None
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                epoch_bar.set_postfix(loss=f"{loss.item():.4f}")

            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["loss"]
            _log(
                f"QA2 Epoch {epoch + 1}/{self.num_epochs} 结束 | "
                f"val_loss={val_loss:.4f}, token_acc={val_metrics['token_acc']:.4f}, "
                f"ppl={val_metrics['ppl']:.4f}, best_loss={self.best_val:.4f}"
            )
            if val_loss < self.best_val:
                self.best_val = val_loss
                path = self.checkpoint_dir / "qa2_best.pt"
                payload = {
                    "model_state_dict": self.model.state_dict(),
                    "config": cfg_base,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                }
                torch.save(payload, path)
                self.best_path = path
                _log(f"QA2 保存最优模型: {path}")
        assert self.best_path is not None
        _log(f"QA2 训练完成 | best_ckpt={self.best_path}, best_val_loss={self.best_val:.4f}")
        return self.best_path
