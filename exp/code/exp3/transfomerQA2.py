import argparse
import json
import math
import os
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

import transformerRaw

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TRAIN = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "..", "..", "data", "SQuAD", "SQuAD-train-v2.0.json")
)
_DEFAULT_DEV = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "..", "..", "data", "SQuAD", "SQuAD-dev-v2.0.json")
)
_DEFAULT_CHECKPOINT_ROOT = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "..", "..", "checkpoint")
)


def _pick_nhead(d_model: int) -> int:
    for h in (12, 8, 6, 4, 2):
        if d_model % h == 0:
            return h
    return 1


def _build_optimizer(name: str, params, lr: float):
    n = name.lower()
    if n == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if n == "adam":
        return torch.optim.Adam(params, lr=lr)
    if n == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"不支持的优化器: {name}，请使用 adamw / adam / sgd")


def _slug_for_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "x"


def _lr_slug(lr: float) -> str:
    return _slug_for_filename(f"{lr:g}")


def best_model_filename(prefix: str, d_model: int, optimizer: str, lr: float, batch_size: int, epoch: int) -> str:
    return (
        f"{prefix}_{d_model}_{_slug_for_filename(optimizer)}_{_lr_slug(lr)}_{batch_size}_{epoch}.pt"
    )


class QADataset(Dataset):
    """SQuAD格式数据处理"""

    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        with open(file_path, encoding="utf-8") as f:
            squad_data = json.load(f)

        for article in squad_data["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    if not qa.get("answers"):
                        continue
                    question = qa["question"]
                    answer = qa["answers"][0]["text"]
                    if not answer.strip():
                        continue

                    input_text = f"[CLS] {question} [SEP] {context} [SEP]"
                    output_text = f"[CLS] {answer} [SEP]"

                    inputs = tokenizer(
                        input_text,
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    targets = tokenizer(
                        output_text,
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )

                    self.data.append(
                        {
                            "input_ids": inputs["input_ids"].squeeze(0),
                            "attention_mask": inputs["attention_mask"].squeeze(0),
                            "labels": targets["input_ids"].squeeze(0),
                        }
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1), :]
        return self.dropout(x)


class TransformerQA(nn.Module):
    """问答模型（含自定义编码器/解码器）"""

    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.encoder = transformerRaw.TransformerEncoder(
            num_encoder_layers, d_model, nhead, dim_feedforward, dropout
        )
        self.decoder = transformerRaw.TransformerDecoder(
            num_decoder_layers, d_model, nhead, dim_feedforward, dropout
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        if memory_mask is None:
            memory_mask = src_mask

        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.encoder(src, src_mask)

        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return self.fc_out(output)

    def generate_mask(self, src, tgt):
        """生成注意力掩码：1 表示保留，0 表示屏蔽（与 MultiHeadAttention.masked_fill 一致）"""
        device = src.device
        b, src_len = src.shape
        _, tgt_len = tgt.shape

        src_mask = (src != 0).view(b, 1, 1, src_len).float()

        subsequent = torch.tril(torch.ones((tgt_len, tgt_len), device=device))
        pad_row = (tgt != 0).float().view(b, 1, tgt_len, 1)
        pad_col = (tgt != 0).float().view(b, 1, 1, tgt_len)
        tgt_mask = pad_row * pad_col * subsequent.view(1, 1, tgt_len, tgt_len)

        return src_mask, tgt_mask

    @torch.no_grad()
    def generate_answer(self, src, tokenizer, device, max_len=64):
        """自回归生成答案 token 序列"""
        self.eval()
        if src.dim() == 1:
            src = src.unsqueeze(0)
        src = src.to(device)

        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        ys = torch.tensor([[cls_id]], dtype=torch.long, device=device)

        for _ in range(max_len):
            src_mask, tgt_mask = self.generate_mask(src, ys)
            logits = self.forward(src, ys, src_mask, tgt_mask, memory_mask=src_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == sep_id:
                break

        return tokenizer.decode(ys[0].tolist(), skip_special_tokens=True)


class QATrainer:
    """训练管理类"""

    def __init__(self, model, tokenizer, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    def prepare_batch(self, batch):
        src = batch["input_ids"].to(self.device)
        tgt = batch["labels"].to(self.device)

        decoder_input = tgt[:, :-1]
        decoder_output = tgt[:, 1:]

        src_mask, tgt_mask = self.model.generate_mask(src, decoder_input)
        return src, decoder_input, decoder_output, src_mask, tgt_mask

    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc="Train", leave=False)
        for batch in pbar:
            src, decoder_in, decoder_out, src_mask, tgt_mask = self.prepare_batch(batch)

            optimizer.zero_grad()
            outputs = self.model(
                src, decoder_in, src_mask, tgt_mask, memory_mask=src_mask
            )

            loss = self.criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                decoder_out.reshape(-1),
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        return total_loss / max(len(dataloader), 1)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc="Eval", leave=False)
        with torch.no_grad():
            for batch in pbar:
                src, decoder_in, decoder_out, src_mask, tgt_mask = self.prepare_batch(
                    batch
                )
                outputs = self.model(
                    src, decoder_in, src_mask, tgt_mask, memory_mask=src_mask
                )
                loss = self.criterion(
                    outputs.reshape(-1, outputs.shape[-1]),
                    decoder_out.reshape(-1),
                )
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        return total_loss / max(len(dataloader), 1)


def main():
    parser = argparse.ArgumentParser(
        description="Transformer Encoder-Decoder SQuAD 训练（自定义 transformerRaw）"
    )
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--train_path", type=str, default=_DEFAULT_TRAIN)
    parser.add_argument("--dev_path", type=str, default=_DEFAULT_DEV)
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=_DEFAULT_CHECKPOINT_ROOT,
    )
    parser.add_argument(
        "--d_model",
        type=int,
        required=True,
        help="词向量 / 模型隐藏维度",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["adamw", "adam", "sgd"],
        help="train 必填",
    )
    parser.add_argument("--lr", type=float, default=None, help="train 必填")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None, help="train 必填")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="eval 模式必填：待加载的 .pt 权重路径",
    )
    args = parser.parse_args()

    if args.mode == "train":
        if args.optimizer is None or args.lr is None or args.epochs is None:
            parser.error("train 模式需要同时指定 --optimizer、--lr、--epochs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("警告: 未检测到 CUDA，将使用 CPU（会很慢）。")
    else:
        print(f"使用设备: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    nhead = _pick_nhead(args.d_model)
    dim_feedforward = 4 * args.d_model

    if args.mode == "train":
        train_path = os.path.normpath(args.train_path)
        dev_path = os.path.normpath(args.dev_path)
        os.makedirs(args.checkpoint_root, exist_ok=True)
        print(f"Checkpoint 目录: {os.path.abspath(args.checkpoint_root)}")

        train_dataset = QADataset(train_path, tokenizer, max_length=args.max_length)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        dev_loader = None
        if os.path.isfile(dev_path):
            dev_dataset = QADataset(dev_path, tokenizer, max_length=args.max_length)
            dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

        model = TransformerQA(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            nhead=nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=args.dropout,
            max_len=args.max_length,
        )
        trainer = QATrainer(model, tokenizer, device=device)
        optimizer = _build_optimizer(args.optimizer, model.parameters(), args.lr)

        best_loss = float("inf")
        best_epoch = -1
        best_path = None

        epoch_bar = tqdm(range(args.epochs), desc="Epochs")
        for epoch in epoch_bar:
            avg_loss = trainer.train_epoch(train_loader, optimizer)
            val_loss = None
            if dev_loader is not None:
                val_loss = trainer.evaluate(dev_loader)
            if val_loss is not None:
                epoch_bar.set_postfix(train_loss=f"{avg_loss:.4f}", dev_loss=f"{val_loss:.4f}")
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | train_loss={avg_loss:.4f} | "
                    f"dev_loss={val_loss:.4f}"
                )
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch + 1
                    fname = best_model_filename(
                        "qa2",
                        args.d_model,
                        args.optimizer,
                        args.lr,
                        args.batch_size,
                        best_epoch,
                    )
                    best_path = os.path.join(args.checkpoint_root, fname)
                    os.makedirs(os.path.dirname(best_path) or ".", exist_ok=True)
                    torch.save(model.state_dict(), best_path)
                    print(f"  ↑ 新的最优模型已保存: {best_path} (dev_loss={best_loss:.4f})")
            else:
                epoch_bar.set_postfix(train_loss=f"{avg_loss:.4f}")
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | train_loss={avg_loss:.4f} | (无 dev，跳过保存)"
                )

        if best_path:
            print(f"训练结束。最优 dev_loss={best_loss:.4f}，模型: {best_path}")
        else:
            print("训练结束：未在 dev 上保存模型（请确认 dev_path 存在）。")

    elif args.mode == "eval":
        if not args.model_path:
            raise ValueError("eval 模式需要 --model_path")
        dev_path = os.path.normpath(args.dev_path)
        if not os.path.isfile(dev_path):
            raise FileNotFoundError(f"未找到验证数据: {dev_path}")
        dev_dataset = QADataset(dev_path, tokenizer, max_length=args.max_length)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
        model = TransformerQA(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            nhead=nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=args.dropout,
            max_len=args.max_length,
        )
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        trainer = QATrainer(model, tokenizer, device=device)
        val_loss = trainer.evaluate(dev_loader)
        print(f"Dev loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
