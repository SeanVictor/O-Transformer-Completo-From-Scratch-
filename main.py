"""
main.py  —  Laboratório 4: O Transformer Completo "From Scratch"
Disciplina: Tópicos em Inteligência Artificial – 2026.1
Instituição: iCEV

Pipeline completo:
    Tarefa 1 → Blocos base: Attention, FFN, Add & Norm
    Tarefa 2 → Encoder Stack (N=6 blocos)
    Tarefa 3 → Decoder Stack (N=6 blocos) + projeção final
    Tarefa 4 → Inferência fim-a-fim com loop auto-regressivo
"""

import numpy as np
from task1_building_blocks import (
    scaled_dot_product_attention,
    FeedForwardNetwork,
    add_and_norm,
    create_causal_mask,
    softmax,
)
from task2_encoder import EncoderStack
from task3_decoder import DecoderStack
from task4_inference import Transformer, run_inference, positional_encoding

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# Hiperparâmetros
# ─────────────────────────────────────────────────────────────
D_MODEL    = 64
N_LAYERS   = 6
VOCAB_SIZE = 10_000

print("=" * 60)
print("  Lab 04 — Transformer Completo 'From Scratch'")
print("=" * 60)
print(f"  d_model    : {D_MODEL}")
print(f"  d_ff       : {D_MODEL * 4}")
print(f"  N camadas  : {N_LAYERS}")
print(f"  Vocab size : {VOCAB_SIZE:,}")


# ════════════════════════════════════════════════════════════
# TAREFA 1 — Verificação dos Blocos Base
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  TAREFA 1 — Blocos Base (Attention, FFN, Add & Norm)")
print("═" * 60)

B, S, D = 1, 5, D_MODEL
X = np.random.randn(B, S, D)

# Attention sem máscara
out_att, _ = scaled_dot_product_attention(X, X, X)
print(f"\n  [Attention sem máscara] {X.shape} → {out_att.shape}")

# Attention com máscara causal
mask = create_causal_mask(S)
out_masked, wm = scaled_dot_product_attention(X, X, X, mask)
assert np.allclose(wm[0][np.triu_indices(S, k=1)], 0.0)
print(f"  [Attention com máscara] {X.shape} → {out_masked.shape}  ✓ futuro=0.0")

# FFN
ffn = FeedForwardNetwork(D)
out_ffn = ffn.forward(X)
print(f"  [FFN]                   {X.shape} → {out_ffn.shape}")

# Add & Norm
out_norm = add_and_norm(X, out_att)
print(f"  [Add & Norm]            {X.shape} → {out_norm.shape}")
print("\n  ✓ Tarefa 1 OK")


# ════════════════════════════════════════════════════════════
# TAREFA 2 — Encoder Stack
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  TAREFA 2 — Encoder Stack (N=6 blocos)")
print("═" * 60)

X_enc = np.random.randn(B, 6, D_MODEL)
encoder = EncoderStack(d_model=D_MODEL, n_layers=N_LAYERS)
Z = encoder.forward(X_enc)

assert Z.shape == X_enc.shape
print(f"\n  ✓ Tarefa 2 OK — shape preservado: {Z.shape}")


# ════════════════════════════════════════════════════════════
# TAREFA 3 — Decoder Stack
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  TAREFA 3 — Decoder Stack (N=6 blocos) + Softmax")
print("═" * 60)

Y_dec   = np.random.randn(B, 4, D_MODEL)
decoder = DecoderStack(d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_layers=N_LAYERS)
probs   = decoder.forward(Y_dec, Z, verbose=True)

assert probs.shape == (B, 4, VOCAB_SIZE)
assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
print(f"\n  ✓ Tarefa 3 OK — probs shape: {probs.shape}, soma por linha ≈ 1.0")


# ════════════════════════════════════════════════════════════
# TAREFA 4 — Inferência Fim-a-Fim
# ════════════════════════════════════════════════════════════
vocab = (
    ["<PAD>", "<START>", "<EOS>", "<UNK>"]
    + ["Thinking", "Machines", "are", "learning", "fast",
       "as", "the", "future", "is", "now"]
    + [f"word_{i}" for i in range(VOCAB_SIZE - 14)]
)

encoder_input_words = ["Thinking", "Machines"]
encoder_token_ids   = [vocab.index(w) for w in encoder_input_words]

model = Transformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS)
run_inference(model, encoder_token_ids, vocab, max_steps=8)

print("\n  ✓ Lab 04 — Pipeline completo executado com sucesso!")
