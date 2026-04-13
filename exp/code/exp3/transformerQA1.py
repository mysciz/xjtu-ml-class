import argparse
import json
import os
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast

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
    """命名: qa1_词向量维度_优化方法_学习率_batch_epoch"""
    return (
        f"{prefix}_{d_model}_{_slug_for_filename(optimizer)}_{_lr_slug(lr)}_{batch_size}_{epoch}.pt"
    )


class Config:
    def __init__(self):
        self.batch_size = 8
        self.learning_rate = 3e-5
        self.epochs = 3
        self.max_length = 384
        self.model_dir = "./model"
        self.train_path = _DEFAULT_TRAIN
        self.dev_path = _DEFAULT_DEV
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = 768
        self.nhead = 12
        self.dim_feedforward = 3072
        self.dropout = 0.1
        self.num_layers = 7
        self.optimizer_name = "adamw"


class SQuADProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def load_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["data"]

    def process(self, data):
        examples = []
        for article in data:
            for para in article["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    if qa.get("is_impossible"):
                        continue
                    if not qa.get("answers"):
                        continue
                    example = {
                        "context": context,
                        "question": qa["question"],
                        "answer": qa["answers"][0],
                    }
                    examples.append(example)
        return examples

    def create_features(self, examples):
        input_ids, masks = [], []
        start_pos, end_pos = [], []

        for ex in examples:
            encoding = self.tokenizer(
                ex["question"],
                ex["context"],
                max_length=self.config.max_length,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True,
            )

            ans_start = ex["answer"]["answer_start"]
            ans_end = ans_start + len(ex["answer"]["text"])

            start_char = ans_start
            end_char = ans_end
            sequence_ids = encoding.sequence_ids()

            start_token, end_token = -1, -1
            for i, (idx, (s, e)) in enumerate(zip(sequence_ids, encoding.offset_mapping)):
                if idx != 1:
                    continue
                if s <= start_char < e:
                    start_token = i
                if s < end_char <= e:
                    end_token = i

            if start_token != -1 and end_token != -1:
                input_ids.append(encoding["input_ids"])
                masks.append(encoding["attention_mask"])
                start_pos.append(start_token)
                end_pos.append(end_token)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(masks),
            "start_pos": torch.tensor(start_pos),
            "end_pos": torch.tensor(end_pos),
        }


class SQuADDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.features.items()}


class TransformerQA(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(config.max_length, config.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.start_fc = nn.Linear(config.d_model, 1)
        self.end_fc = nn.Linear(config.d_model, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids) + self.pos_encoder[: input_ids.size(1)]
        x = self.dropout(x)
        x = x.permute(1, 0, 2)

        output = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        output = output.permute(1, 0, 2)

        start_logits = self.start_fc(output).squeeze(-1)
        end_logits = self.end_fc(output).squeeze(-1)
        return start_logits, end_logits


class QATrainer:
    def __init__(self, config, model, train_loader, dev_loader=None):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.optimizer = _build_optimizer(
            config.optimizer_name, model.parameters(), config.learning_rate
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.config.device)
            mask = batch["attention_mask"].to(self.config.device)
            start = batch["start_pos"].to(self.config.device)
            end = batch["end_pos"].to(self.config.device)

            self.optimizer.zero_grad()
            s_logits, e_logits = self.model(input_ids, mask)

            loss = nn.CrossEntropyLoss()(s_logits, start) + nn.CrossEntropyLoss()(
                e_logits, end
            )

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        return total_loss / max(len(self.train_loader), 1)

    def evaluate(self):
        if self.dev_loader is None:
            return {}
        self.model.eval()
        correct = 0
        total = 0
        pbar = tqdm(self.dev_loader, desc="Eval", leave=False)
        with torch.no_grad():
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.config.device)
                mask = batch["attention_mask"].to(self.config.device)
                start = batch["start_pos"].to(self.config.device)
                end = batch["end_pos"].to(self.config.device)
                s_logits, e_logits = self.model(input_ids, mask)
                pred_start = s_logits.argmax(dim=-1)
                pred_end = e_logits.argmax(dim=-1)
                correct += ((pred_start == start) & (pred_end == end)).sum().item()
                total += start.size(0)
        return {"span_exact_acc": correct / total if total else 0.0}

    def save_model(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.config.device))


def main():
    parser = argparse.ArgumentParser(description="Transformer QA1（抽取式）SQuAD 训练/测试")
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--train_path", type=str, default=_DEFAULT_TRAIN)
    parser.add_argument("--dev_path", type=str, default=_DEFAULT_DEV)
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=_DEFAULT_CHECKPOINT_ROOT,
        help="最优模型保存目录（文件名自带超参，不再套子目录）",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        required=True,
        help="词向量 / 模型维度（与嵌入维一致）",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["adamw", "adam", "sgd"],
        help="优化方法（train 必填）",
    )
    parser.add_argument("--lr", type=float, default=None, help="学习率（train 必填）")
    parser.add_argument("--batch_size", type=int, required=True, help="batch 大小")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数；文件名中的 epoch 为验证最优时所在轮（train 必填）",
    )
    args = parser.parse_args()

    if args.mode == "train":
        if args.optimizer is None or args.lr is None or args.epochs is None:
            parser.error("train 模式需要同时指定 --optimizer、--lr、--epochs")

    config = Config()
    config.train_path = os.path.normpath(args.train_path)
    config.dev_path = os.path.normpath(args.dev_path)
    config.d_model = args.d_model
    config.nhead = _pick_nhead(config.d_model)
    config.dim_feedforward = 4 * config.d_model
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.optimizer_name = args.optimizer

    if not torch.cuda.is_available():
        print("警告: 未检测到 CUDA，将使用 CPU（会很慢）。")
    else:
        print(f"使用设备: {config.device}")

    processor = SQuADProcessor(config)

    if args.mode == "train":
        os.makedirs(args.checkpoint_root, exist_ok=True)
        print(f"Checkpoint 目录: {os.path.abspath(args.checkpoint_root)}")

        train_data = processor.load_data(config.train_path)
        train_examples = processor.process(train_data)
        train_features = processor.create_features(train_examples)
        train_dataset = SQuADDataset(train_features)
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )

        dev_data = processor.load_data(config.dev_path)
        dev_examples = processor.process(dev_data)
        dev_features = processor.create_features(dev_examples)
        dev_loader = DataLoader(
            SQuADDataset(dev_features), batch_size=config.batch_size
        )

        model = TransformerQA(config, processor.tokenizer.vocab_size)
        trainer = QATrainer(config, model, train_loader, dev_loader)

        best_acc = -1.0
        best_epoch = -1
        best_path = None

        epoch_bar = tqdm(range(config.epochs), desc="Epochs")
        for epoch in epoch_bar:
            avg_loss = trainer.train_epoch()
            metrics = trainer.evaluate()
            acc = metrics.get("span_exact_acc", 0.0)
            epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", span_acc=f"{acc:.4f}")
            print(
                f"Epoch {epoch + 1}/{config.epochs} | train_loss={avg_loss:.4f} | "
                f"dev_span_exact_acc={acc:.4f}"
            )

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch + 1
                fname = best_model_filename(
                    "qa1",
                    config.d_model,
                    config.optimizer_name,
                    config.learning_rate,
                    config.batch_size,
                    best_epoch,
                )
                best_path = os.path.join(args.checkpoint_root, fname)
                trainer.save_model(best_path)
                print(f"  ↑ 新的最优模型已保存: {best_path} (span_exact_acc={best_acc:.4f})")

        if best_path:
            print(f"训练结束。最优 span_exact_acc={best_acc:.4f}，模型: {best_path}")
        else:
            print("训练结束，但未产生有效验证指标（请检查数据与 batch）。")

    elif args.mode == "test":
        if not args.model_path:
            raise ValueError("test 模式需要 --model_path")
        dev_data = processor.load_data(config.dev_path)
        dev_examples = processor.process(dev_data)
        dev_features = processor.create_features(dev_examples)
        dev_loader = DataLoader(
            SQuADDataset(dev_features), batch_size=config.batch_size
        )

        model = TransformerQA(config, processor.tokenizer.vocab_size)
        trainer = QATrainer(config, model, None, dev_loader)
        trainer.load_model(args.model_path)
        metrics = trainer.evaluate()
        print(f"Test Results - {metrics}")


if __name__ == "__main__":
    main()
