"""
Tarefa 2: Montando a Pilha do Encoder

Fluxo por bloco:
    1. Self-Attention(X)        → Q, K, V gerados a partir de X
    2. Add & Norm               → X = LayerNorm(X + att_out)
    3. FFN(X)
    4. Add & Norm               → X = LayerNorm(X + ffn_out)

N blocos empilhados → saída Z (memória rica contextualizada)
"""

import numpy as np
from task1_building_blocks import (
    scaled_dot_product_attention,
    FeedForwardNetwork,
    add_and_norm,
)


# ─────────────────────────────────────────────────────────────
# Projeções lineares para Q, K, V
# ─────────────────────────────────────────────────────────────

class LinearProjection:
    """Camada linear simples: Y = X @ W"""
    def __init__(self, d_in, d_out):
        self.W = np.random.randn(d_in, d_out) * np.sqrt(2.0 / d_in)

    def forward(self, X):
        return X @ self.W


# ─────────────────────────────────────────────────────────────
# Bloco único do Encoder
# ─────────────────────────────────────────────────────────────

class EncoderBlock:
    """
    Um bloco do Encoder:
        Sub-camada 1 → Self-Attention (sem máscara) + Add & Norm
        Sub-camada 2 → FFN                          + Add & Norm
    """

    def __init__(self, d_model, d_ff=None):
        self.d_model = d_model
        # Projeções para Self-Attention
        self.W_Q = LinearProjection(d_model, d_model)
        self.W_K = LinearProjection(d_model, d_model)
        self.W_V = LinearProjection(d_model, d_model)
        # FFN
        self.ffn = FeedForwardNetwork(d_model, d_ff)

    def forward(self, X):
        """
        X : (batch, seq, d_model)
        Retorna tensor de mesma shape com representações contextualizadas.
        """
        # ── Sub-camada 1: Self-Attention ──────────────────────
        Q = self.W_Q.forward(X)
        K = self.W_K.forward(X)
        V = self.W_V.forward(X)

        att_out, _ = scaled_dot_product_attention(Q, K, V, mask=None)
        X = add_and_norm(X, att_out)

        # ── Sub-camada 2: FFN ─────────────────────────────────
        ffn_out = self.ffn.forward(X)
        X = add_and_norm(X, ffn_out)

        return X


# ─────────────────────────────────────────────────────────────
# Pilha do Encoder (N blocos)
# ─────────────────────────────────────────────────────────────

class EncoderStack:
    """
    Empilha N blocos idênticos do Encoder.
    Cada bloco tem seus próprios pesos independentes.
    """

    def __init__(self, d_model, n_layers=6, d_ff=None):
        self.blocks = [EncoderBlock(d_model, d_ff) for _ in range(n_layers)]
        self.n_layers = n_layers

    def forward(self, X):
        """
        X : (batch, seq, d_model)  — entrada com Positional Encoding
        Z : (batch, seq, d_model)  — memória rica contextualizada
        """
        print(f"\n  [Encoder] input  : {X.shape}")
        for i, block in enumerate(self.blocks, 1):
            X = block.forward(X)
            print(f"  [Encoder] camada {i}/{self.n_layers} → {X.shape}")
        print(f"  [Encoder] output Z : {X.shape}  ✓")
        return X   # Z


# ─────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    B, S, D = 1, 6, 64

    X = np.random.randn(B, S, D)   # simula embedding + positional encoding

    encoder = EncoderStack(d_model=D, n_layers=6)
    Z = encoder.forward(X)

    assert Z.shape == X.shape, f"ERRO: {Z.shape} ≠ {X.shape}"
    print("\n  ✓ Tarefa 2 — Encoder Stack OK")
