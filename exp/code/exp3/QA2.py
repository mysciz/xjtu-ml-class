from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from Transformer import TransformerDecoder, TransformerEncoder, build_causal_mask


def _log(msg: str) -> None:
    print(f"[QA2] {msg}")


class GenerativeQA2(nn.Module):
    """生成式问答：手写 BERT 风格 Encoder-Decoder。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
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
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.token_type_embeddings = nn.Embedding(2, d_model)
        self.emb_layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.emb_dropout = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, nhead, dim_feedforward, dropout)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.word_embeddings.weight
        if use_pretrained_init:
            ok = self.init_from_bert(pretrained_name=pretrained_name, local_files_only=local_files_only)
            if ok:
                _log("已完成 BERT 权重初始化。")
            else:
                _log("BERT 权重初始化失败，已回退到随机初始化继续训练。")

    def _embed(self, ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(ids)
        pos_ids = torch.arange(ids.size(1), device=ids.device).unsqueeze(0).expand_as(ids)
        x = (
            self.word_embeddings(ids)
            + self.position_embeddings(pos_ids.clamp(max=self.max_len - 1))
            + self.token_type_embeddings(token_type_ids.clamp(min=0, max=1))
        )
        return self.emb_dropout(self.emb_layer_norm(x))

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

        n_enc = min(len(self.encoder.layers), len(bert.encoder.layer))
        n_dec = min(len(self.decoder.layers), len(bert.encoder.layer))

        for i in range(n_enc):
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

        for i in range(n_dec):
            src = bert.encoder.layer[i]
            dst = self.decoder.layers[i]
            # self-attn
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
            # cross-attn 用同层 self-attn 初始化
            dst.cross_attn.q_proj.weight.copy_(src.attention.self.query.weight)
            dst.cross_attn.q_proj.bias.copy_(src.attention.self.query.bias)
            dst.cross_attn.k_proj.weight.copy_(src.attention.self.key.weight)
            dst.cross_attn.k_proj.bias.copy_(src.attention.self.key.bias)
            dst.cross_attn.v_proj.weight.copy_(src.attention.self.value.weight)
            dst.cross_attn.v_proj.bias.copy_(src.attention.self.value.bias)
            dst.cross_attn.out_proj.weight.copy_(src.attention.output.dense.weight)
            dst.cross_attn.out_proj.bias.copy_(src.attention.output.dense.bias)
            dst.norm2.weight.copy_(src.attention.output.LayerNorm.weight)
            dst.norm2.bias.copy_(src.attention.output.LayerNorm.bias)
            # ffn
            dst.ffn.linear1.weight.copy_(src.intermediate.dense.weight)
            dst.ffn.linear1.bias.copy_(src.intermediate.dense.bias)
            dst.ffn.linear2.weight.copy_(src.output.dense.weight)
            dst.ffn.linear2.bias.copy_(src.output.dense.bias)
            dst.norm3.weight.copy_(src.output.LayerNorm.weight)
            dst.norm3.bias.copy_(src.output.LayerNorm.bias)
        return True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        src_pad = attention_mask == 0 if attention_mask is not None else None
        x = self._embed(input_ids)
        mem = self.encoder(x, key_padding_mask=src_pad)

        if decoder_input_ids is None:
            return mem, None

        y = self._embed(decoder_input_ids)
        lt = y.size(1)
        causal = build_causal_mask(lt, y.device, y.dtype)
        tgt_pad = decoder_input_ids == 0 if decoder_input_ids is not None else None
        h = self.decoder(y, mem, causal, memory_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
        logits = self.lm_head(h)
        loss = None
        if decoder_labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), decoder_labels.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def greedy_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_new_tokens: int,
        pad_token_id: int,
        cls_token_id: int,
        sep_token_id: int,
        min_new_tokens: int = 2,
        repetition_penalty: float = 1.1,
    ) -> torch.Tensor:
        self.eval()
        src_pad = attention_mask == 0 if attention_mask is not None else None
        x = self._embed(input_ids)
        mem = self.encoder(x, key_padding_mask=src_pad)

        ys = torch.full((input_ids.size(0), 1), cls_token_id, dtype=torch.long, device=input_ids.device)
        for step in range(max_new_tokens):
            y = self._embed(ys)
            lt = y.size(1)
            causal = build_causal_mask(lt, y.device, y.dtype)
            tgt_pad = ys == pad_token_id
            h = self.decoder(y, mem, causal, memory_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
            next_logits = self.lm_head(h[:, -1, :])
            if repetition_penalty > 1.0:
                for b in range(ys.size(0)):
                    used = torch.unique(ys[b])
                    next_logits[b, used] = next_logits[b, used] / repetition_penalty
            if step < min_new_tokens:
                next_logits[:, sep_token_id] = -1e4
            nxt = torch.argmax(next_logits, dim=-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
            if torch.all(nxt.squeeze(-1) == sep_token_id):
                break
        return ys
