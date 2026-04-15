from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertForQuestionAnswering, BertModel

from Transformer import TransformerEncoder


def _log(msg: str) -> None:
    print(f"[QA1] {msg}")


class ExtractiveQA1(nn.Module):
    """抽取式问答：手写 BERT 风格编码器 + QA 头。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        max_len: int = 512,
        use_pretrained_init: bool = False,
        pretrained_name: str = "bert-base-uncased",
        local_files_only: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.token_type_embeddings = nn.Embedding(2, d_model)
        self.emb_layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.emb_dropout = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, nhead, dim_feedforward, dropout)
        self.start_head = nn.Linear(d_model, 1)
        self.end_head = nn.Linear(d_model, 1)
        if use_pretrained_init:
            ok = self.init_from_bert(pretrained_name=pretrained_name, local_files_only=local_files_only)
            if ok:
                _log("已完成 BERT 权重初始化。")
            else:
                _log("BERT 权重初始化失败，已回退到随机初始化继续训练。")

    @torch.no_grad()
    def init_from_bert(self, pretrained_name: str = "bert-base-uncased", local_files_only: bool = False) -> bool:
        try:
            bert = BertModel.from_pretrained(pretrained_name, local_files_only=local_files_only)
        except Exception as e:
            _log(f"加载 BERT 编码器失败: {e}")
            return False
        self.word_embeddings.weight.copy_(bert.embeddings.word_embeddings.weight[: self.word_embeddings.num_embeddings])
        self.position_embeddings.weight.copy_(bert.embeddings.position_embeddings.weight[: self.max_len])
        self.token_type_embeddings.weight.copy_(bert.embeddings.token_type_embeddings.weight[:2])
        self.emb_layer_norm.weight.copy_(bert.embeddings.LayerNorm.weight)
        self.emb_layer_norm.bias.copy_(bert.embeddings.LayerNorm.bias)

        n = min(len(self.encoder.layers), len(bert.encoder.layer))
        for i in range(n):
            src = bert.encoder.layer[i]
            dst = self.encoder.layers[i]
            dst.self_attn.q_proj.weight.copy_(src.attention.self.query.weight)
            dst.self_attn.q_proj.bias.copy_(src.attention.self.query.bias)
            dst.self_attn.k_proj.weight.copy_(src.attention.self.key.weight)
            dst.self_attn.k_proj.bias.copy_(src.attention.self.key.bias)
            dst.self_attn.v_proj.weight.copy_(src.attention.self.value.weight)
            dst.self_attn.v_proj.bias.copy_(src.attention.self.value.bias)
            dst.self_attn.out_proj.weight.copy_(src.attention.output.dense.weight)
            dst.self_attn.out_proj.bias.copy_(src.attention.output.dense.bias)
            dst.norm1.weight.copy_(src.attention.output.LayerNorm.weight)
            dst.norm1.bias.copy_(src.attention.output.LayerNorm.bias)
            dst.ffn.linear1.weight.copy_(src.intermediate.dense.weight)
            dst.ffn.linear1.bias.copy_(src.intermediate.dense.bias)
            dst.ffn.linear2.weight.copy_(src.output.dense.weight)
            dst.ffn.linear2.bias.copy_(src.output.dense.bias)
            dst.norm2.weight.copy_(src.output.LayerNorm.weight)
            dst.norm2.bias.copy_(src.output.LayerNorm.bias)

        # 若隐藏维度一致，顺便初始化 QA 头
        if self.d_model == bert.config.hidden_size:
            try:
                qa = BertForQuestionAnswering.from_pretrained(pretrained_name, local_files_only=local_files_only)
                self.start_head.weight.copy_(qa.qa_outputs.weight[:1, :])
                self.start_head.bias.copy_(qa.qa_outputs.bias[:1])
                self.end_head.weight.copy_(qa.qa_outputs.weight[1:2, :])
                self.end_head.bias.copy_(qa.qa_outputs.bias[1:2])
            except Exception as e:
                _log(f"加载 QA 头权重失败（仅头部随机初始化）: {e}")
        return True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        pos_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(pos_ids.clamp(max=self.max_len - 1))
            + self.token_type_embeddings(token_type_ids.clamp(min=0, max=1))
        )
        x = self.emb_dropout(self.emb_layer_norm(x))
        key_padding = attention_mask == 0 if attention_mask is not None else None
        h = self.encoder(x, key_padding_mask=key_padding)
        start_logits = self.start_head(h).squeeze(-1)
        end_logits = self.end_head(h).squeeze(-1)
        if attention_mask is not None:
            start_logits = start_logits.masked_fill(attention_mask == 0, -1e4)
            end_logits = end_logits.masked_fill(attention_mask == 0, -1e4)
        loss = None
        if start_positions is not None and end_positions is not None:
            loss_f = nn.CrossEntropyLoss()
            loss = loss_f(start_logits, start_positions) + loss_f(end_logits, end_positions)
        return start_logits, end_logits, loss
