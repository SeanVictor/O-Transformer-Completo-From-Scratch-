"""
Tarefa 1: Refatoração e Integração — Os Blocos de Montar

Reúne em um único módulo os componentes matemáticos dos Labs anteriores:
  - scaled_dot_product_attention(Q, K, V, mask)
  - FeedForwardNetwork  (FFN)
  - add_and_norm        (conexão residual + LayerNorm)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────

def softmax(x):
    """Softmax numericamente estável sobre o último eixo."""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x_shifted)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def create_causal_mask(seq_len):
    """
    Máscara causal [seq_len, seq_len]:
      triangular inferior + diagonal → 0
      triangular superior            → -inf
    """
    upper = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    return np.where(upper, -np.inf, 0.0)


# ─────────────────────────────────────────────────────────────
# 1. Scaled Dot-Product Attention
# ─────────────────────────────────────────────────────────────

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) + mask ) V

    Parâmetros:
        Q, K, V : (batch, seq, d_model)
        mask    : (seq_q, seq_k) opcional — máscara causal ou None

    Retorna:
        output       : (batch, seq_q, d_model)
        attn_weights : (batch, seq_q, seq_k)
    """
    d_k = Q.shape[-1]

    # Produto escalar escalado
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)   # (batch, seq_q, seq_k)

    # Aplica máscara se fornecida
    if mask is not None:
        scores = scores + mask

    attn_weights = softmax(scores)
    output = attn_weights @ V
    return output, attn_weights


# ─────────────────────────────────────────────────────────────
# 2. Feed-Forward Network (FFN)
# ─────────────────────────────────────────────────────────────

class FeedForwardNetwork:
    """
    FFN(x) = max(0, x·W1 + b1)·W2 + b2

    Expande d_model → d_ff (4×) com ReLU e contrai de volta.
    """

    def __init__(self, d_model, d_ff=None):
        self.d_ff = d_ff or d_model * 4
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / self.d_ff)
        self.W1 = np.random.randn(d_model, self.d_ff)    * scale1
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model)    * scale2
        self.b2 = np.zeros(d_model)

    def forward(self, X):
        """X: (batch, seq, d_model) → (batch, seq, d_model)"""
        hidden = np.maximum(0, X @ self.W1 + self.b1)   # ReLU
        return hidden @ self.W2 + self.b2


# ─────────────────────────────────────────────────────────────
# 3. Add & Norm
# ─────────────────────────────────────────────────────────────

def layer_norm(X, eps=1e-6):
    """LayerNorm sobre o último eixo (features)."""
    mean = np.mean(X, axis=-1, keepdims=True)
    var  = np.var(X,  axis=-1, keepdims=True)
    return (X - mean) / np.sqrt(var + eps)


def add_and_norm(X, sublayer_output, eps=1e-6):
    """
    Output = LayerNorm(X + Sublayer(X))
    Conexão residual seguida de normalização de camada.
    """
    return layer_norm(X + sublayer_output, eps)


# ─────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    B, S, D = 1, 5, 64

    X = np.random.randn(B, S, D)

    # Atenção sem máscara
    out_att, w = scaled_dot_product_attention(X, X, X)
    print(f"[Attention sem máscara] {X.shape} → {out_att.shape}")

    # Atenção com máscara causal
    mask = create_causal_mask(S)
    out_masked, wm = scaled_dot_product_attention(X, X, X, mask)
    print(f"[Attention com máscara] {X.shape} → {out_masked.shape}")
    assert np.allclose(wm[0][np.triu_indices(S, k=1)], 0.0)
    print("  ✓ Máscara causal OK — posições futuras = 0.0")

    # FFN
    ffn = FeedForwardNetwork(D)
    out_ffn = ffn.forward(X)
    print(f"[FFN]                   {X.shape} → {out_ffn.shape}")

    # Add & Norm
    out_norm = add_and_norm(X, out_att)
    print(f"[Add & Norm]            {X.shape} → {out_norm.shape}")

    print("\n  ✓ Tarefa 1 — todos os blocos operacionais")
