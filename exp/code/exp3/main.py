from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_TRAIN = REPO_ROOT / "exp" / "data" / "SQuAD" / "SQuAD-train-v2.0.json"
DATA_DEV = REPO_ROOT / "exp" / "data" / "SQuAD" / "SQuAD-dev-v2.0.json"
CHECKPOINT_DIR = REPO_ROOT / "exp" / "checkpoint"

TOKENIZER_NAME = "bert-base-uncased"
USE_PRETRAINED_INIT = True
LOCAL_FILES_ONLY = False

# 训练器对外关键超参：batch_size、优化方法、学习率、词向量维度（即 d_model）
BATCH_SIZE = 8
OPTIMIZER = "adamw"
LEARNING_RATE = 3e-4
EMBEDDING_DIM = 768

NUM_EPOCHS = 3
NHEAD = 12
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 3072
DROPOUT = 0.1
MAX_EXTRACT_LEN = 256
MAX_ENC_LEN = 256
MAX_DEC_LEN = 96
MAX_TRAIN_SAMPLES =5000
MAX_VAL_SAMPLES = 800


def _log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}][Main] {msg}")


def main() -> None:
    from transformers import BertTokenizerFast

    from Dataset import SquadExtractiveDataset, SquadGenerativeDataset
    from QA1 import ExtractiveQA1
    from QA2 import GenerativeQA2
    from Train import QA1Trainer, QA2Trainer
    from infer import InferenceRunner

    _log("程序启动")
    _log(
        f"配置: tokenizer={TOKENIZER_NAME}, batch_size={BATCH_SIZE}, optimizer={OPTIMIZER}, "
        f"lr={LEARNING_RATE}, embedding_dim={EMBEDDING_DIM}, epochs={NUM_EPOCHS}, "
        f"use_pretrained_init={USE_PRETRAINED_INIT}"
    )
    _log(f"数据路径: train={DATA_TRAIN}, dev={DATA_DEV}")
    _log(f"checkpoint路径: {CHECKPOINT_DIR}")

    _log("加载 tokenizer")
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_NAME)
    vocab_size = tokenizer.vocab_size
    _log(f"tokenizer 加载完成 | vocab_size={vocab_size}")

    _log("构建抽取式训练/验证数据集")
    train_ext = SquadExtractiveDataset(
        DATA_TRAIN,
        tokenizer_name=TOKENIZER_NAME,
        max_length=MAX_EXTRACT_LEN,
        max_samples=MAX_TRAIN_SAMPLES,
    )
    val_ext = SquadExtractiveDataset(
        DATA_DEV,
        tokenizer_name=TOKENIZER_NAME,
        max_length=MAX_EXTRACT_LEN,
        max_samples=MAX_VAL_SAMPLES,
    )
    _log(f"抽取式数据集就绪 | train={len(train_ext)}, val={len(val_ext)}")

    _log("构建 QA1 模型（手写 Transformer）")
    model_qa1 = ExtractiveQA1(
        vocab_size=vocab_size,
        d_model=EMBEDDING_DIM,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=512,
        use_pretrained_init=USE_PRETRAINED_INIT,
        pretrained_name=TOKENIZER_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    )
    cfg_qa1 = {
        "vocab_size": vocab_size,
        "d_model": EMBEDDING_DIM,
        "nhead": NHEAD,
        "num_encoder_layers": NUM_ENCODER_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "max_len": 512,
        "use_pretrained_init": USE_PRETRAINED_INIT,
        "pretrained_name": TOKENIZER_NAME,
        "local_files_only": LOCAL_FILES_ONLY,
    }

    _log("开始训练 QA1")
    trainer1 = QA1Trainer(
        model_qa1,
        train_ext,
        val_ext,
        batch_size=BATCH_SIZE,
        optimizer_name=OPTIMIZER,
        learning_rate=LEARNING_RATE,
        embedding_dim=EMBEDDING_DIM,
        num_epochs=NUM_EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR,
    )
    ckpt_qa1 = trainer1.fit(TOKENIZER_NAME, extra_config=cfg_qa1)
    _log(f"QA1 训练完成 | best_ckpt={ckpt_qa1}")

    _log("构建生成式训练/验证数据集")
    train_gen = SquadGenerativeDataset(
        DATA_TRAIN,
        tokenizer_name=TOKENIZER_NAME,
        max_enc_length=MAX_ENC_LEN,
        max_dec_length=MAX_DEC_LEN,
        max_samples=MAX_TRAIN_SAMPLES,
    )
    val_gen = SquadGenerativeDataset(
        DATA_DEV,
        tokenizer_name=TOKENIZER_NAME,
        max_enc_length=MAX_ENC_LEN,
        max_dec_length=MAX_DEC_LEN,
        max_samples=MAX_VAL_SAMPLES,
    )
    _log(f"生成式数据集就绪 | train={len(train_gen)}, val={len(val_gen)}")

    _log("构建 QA2 模型（手写 Transformer）")
    model_qa2 = GenerativeQA2(
        vocab_size=vocab_size,
        d_model=EMBEDDING_DIM,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=512,
        use_pretrained_init=USE_PRETRAINED_INIT,
        pretrained_name=TOKENIZER_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    )
    cfg_qa2 = {
        **cfg_qa1,
        "num_decoder_layers": NUM_DECODER_LAYERS,
        "max_dec_length": MAX_DEC_LEN,
    }

    _log("开始训练 QA2")
    trainer2 = QA2Trainer(
        model_qa2,
        train_gen,
        val_gen,
        batch_size=BATCH_SIZE,
        optimizer_name=OPTIMIZER,
        learning_rate=LEARNING_RATE,
        embedding_dim=EMBEDDING_DIM,
        num_epochs=NUM_EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR,
    )
    ckpt_qa2 = trainer2.fit(TOKENIZER_NAME, extra_config=cfg_qa2)
    _log(f"QA2 训练完成 | best_ckpt={ckpt_qa2}")

    _log("开始随机抽样推理并写日志")
    runner = InferenceRunner(checkpoint_dir=CHECKPOINT_DIR)
    runner.run_qa1_random(
        ckpt_qa1,
        DATA_DEV,
        num_samples=5,
        pool_size=2000,
        max_length=MAX_EXTRACT_LEN,
        max_answer_len=30,
    )
    runner.run_qa2_random(
        ckpt_qa2,
        DATA_DEV,
        num_samples=5,
        pool_size=2000,
        max_enc_length=MAX_ENC_LEN,
        max_new_tokens=48,
        min_new_tokens=2,
        repetition_penalty=1.1,
    )
    _log("全部流程结束")


if __name__ == "__main__":
    main()
