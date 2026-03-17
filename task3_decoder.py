"""
Tarefa 3: Montando a Pilha do Decoder

Fluxo por bloco:
    1. Masked Self-Attention(Y)     — máscara causal (-inf no futuro)
    2. Add & Norm
    3. Cross-Attention(Q=Y, K=Z, V=Z) — ponte Encoder-Decoder
    4. Add & Norm
    5. FFN
    6. Add & Norm

Saída final → projeção Linear → Softmax → probabilidades do vocabulário
"""

import numpy as np
from task1_building_blocks import (
    scaled_dot_product_attention,
    FeedForwardNetwork,
    add_and_norm,
    softmax,
    create_causal_mask,
)
from task2_encoder import LinearProjection


# ─────────────────────────────────────────────────────────────
# Bloco único do Decoder
# ─────────────────────────────────────────────────────────────

class DecoderBlock:
    """
    Um bloco do Decoder:
        Sub-camada 1 → Masked Self-Attention + Add & Norm
        Sub-camada 2 → Cross-Attention       + Add & Norm
        Sub-camada 3 → FFN                   + Add & Norm
    """

    def __init__(self, d_model, d_ff=None):
        self.d_model = d_model

        # Pesos para Masked Self-Attention
        self.WQ_self = LinearProjection(d_model, d_model)
        self.WK_self = LinearProjection(d_model, d_model)
        self.WV_self = LinearProjection(d_model, d_model)

        # Pesos para Cross-Attention
        self.WQ_cross = LinearProjection(d_model, d_model)
        self.WK_cross = LinearProjection(d_model, d_model)
        self.WV_cross = LinearProjection(d_model, d_model)

        # FFN
        self.ffn = FeedForwardNetwork(d_model, d_ff)

    def forward(self, Y, Z):
        """
        Y : (batch, seq_dec, d_model)  — tokens gerados pelo Decoder
        Z : (batch, seq_enc, d_model)  — memória do Encoder

        Retorna: (batch, seq_dec, d_model)
        """
        seq_dec = Y.shape[1]

        # ── Sub-camada 1: Masked Self-Attention ───────────────
        mask = create_causal_mask(seq_dec)   # (seq_dec, seq_dec)

        Q = self.WQ_self.forward(Y)
        K = self.WK_self.forward(Y)
        V = self.WV_self.forward(Y)

        att_out, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
        Y = add_and_norm(Y, att_out)

        # ── Sub-camada 2: Cross-Attention ─────────────────────
        Q_c = self.WQ_cross.forward(Y)   # Q vem do Decoder
        K_c = self.WK_cross.forward(Z)   # K vem do Encoder
        V_c = self.WV_cross.forward(Z)   # V vem do Encoder

        cross_out, _ = scaled_dot_product_attention(Q_c, K_c, V_c, mask=None)
        Y = add_and_norm(Y, cross_out)

        # ── Sub-camada 3: FFN ─────────────────────────────────
        ffn_out = self.ffn.forward(Y)
        Y = add_and_norm(Y, ffn_out)

        return Y


# ─────────────────────────────────────────────────────────────
# Pilha do Decoder (N blocos) + cabeça de projeção
# ─────────────────────────────────────────────────────────────

class DecoderStack:
    """
    N blocos de DecoderBlock empilhados, seguidos de:
        → Camada Linear  (d_model → vocab_size)
        → Softmax        → distribuição de probabilidades
    """

    def __init__(self, d_model, vocab_size, n_layers=6, d_ff=None):
        self.blocks     = [DecoderBlock(d_model, d_ff) for _ in range(n_layers)]
        self.n_layers   = n_layers
        self.vocab_size = vocab_size
        # Projeção final: d_model → vocab_size
        self.W_out = np.random.randn(d_model, vocab_size) * np.sqrt(2.0 / d_model)

    def forward(self, Y, Z, verbose=False):
        """
        Y    : (batch, seq_dec, d_model)
        Z    : (batch, seq_enc, d_model)
        verbose : imprime shapes por camada

        Retorna:
            probs : (batch, seq_dec, vocab_size) — distribuição Softmax
        """
        if verbose:
            print(f"\n  [Decoder] input  : {Y.shape}")

        for i, block in enumerate(self.blocks, 1):
            Y = block.forward(Y, Z)
            if verbose:
                print(f"  [Decoder] camada {i}/{self.n_layers} → {Y.shape}")

        # Projeção linear → vocabulário
        logits = Y @ self.W_out                       # (batch, seq_dec, vocab_size)
        probs  = softmax(logits)                      # (batch, seq_dec, vocab_size)

        if verbose:
            print(f"  [Decoder] logits : {logits.shape}")
            print(f"  [Decoder] probs  : {probs.shape}  ✓")

        return probs


# ─────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    B, S_ENC, S_DEC, D = 1, 6, 4, 64
    VOCAB = 10_000

    Z = np.random.randn(B, S_ENC, D)   # saída fictícia do Encoder
    Y = np.random.randn(B, S_DEC, D)   # tokens fictícios do Decoder

    decoder = DecoderStack(d_model=D, vocab_size=VOCAB, n_layers=6)
    probs   = decoder.forward(Y, Z, verbose=True)

    assert probs.shape == (B, S_DEC, VOCAB)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
    print("\n  ✓ Tarefa 3 — Decoder Stack OK")
