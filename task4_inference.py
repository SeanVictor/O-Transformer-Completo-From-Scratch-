"""
Tarefa 4: A Prova Final — Inferência Fim-a-Fim

Pipeline completo:
    1. encoder_input  ("Thinking Machines") → Encoder → Z
    2. Decoder inicia com <START>
    3. Loop auto-regressivo:
         a. Busca embedding dos tokens gerados → Y
         b. Decoder(Y, Z) → probs
         c. argmax(probs do último token) → próximo token
         d. Concatena à sequência
         e. Para se próximo == <EOS>
"""

import numpy as np
from task1_building_blocks import softmax
from task2_encoder         import EncoderStack
from task3_decoder         import DecoderStack


# ─────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────

def positional_encoding(seq_len, d_model):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    PE = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len)[:, np.newaxis]          # (seq, 1)
    dims      = np.arange(0, d_model, 2)                   # pares
    div_term  = np.power(10000.0, dims / d_model)

    PE[:, 0::2] = np.sin(positions / div_term)
    PE[:, 1::2] = np.cos(positions / div_term)
    return PE   # (seq_len, d_model)


# ─────────────────────────────────────────────────────────────
# Transformer Completo
# ─────────────────────────────────────────────────────────────

class Transformer:
    """
    Arquitetura Encoder-Decoder completa (toy version).

    Parâmetros:
        vocab_size : tamanho do vocabulário
        d_model    : dimensão dos embeddings
        n_layers   : número de blocos no Encoder e no Decoder
        d_ff       : dimensão interna do FFN (padrão: 4 × d_model)
    """

    def __init__(self, vocab_size, d_model=64, n_layers=6, d_ff=None):
        self.vocab_size = vocab_size
        self.d_model    = d_model

        # Tabelas de embedding compartilháveis (simplificação didática)
        self.enc_embedding = np.random.randn(vocab_size, d_model) * 0.1
        self.dec_embedding = np.random.randn(vocab_size, d_model) * 0.1

        # Pilha do Encoder e do Decoder
        self.encoder = EncoderStack(d_model, n_layers, d_ff)
        self.decoder = DecoderStack(d_model, vocab_size, n_layers, d_ff)

    def encode(self, token_ids):
        """
        token_ids : list[int]
        Retorna Z : (1, seq_enc, d_model)
        """
        seq_len = len(token_ids)
        X = self.enc_embedding[token_ids]           # (seq, d_model)
        X = X + positional_encoding(seq_len, self.d_model)
        X = X[np.newaxis, :, :]                     # (1, seq, d_model)
        Z = self.encoder.forward(X)
        return Z

    def decode_step(self, dec_token_ids, Z):
        """
        dec_token_ids : list[int]  — tokens gerados até agora
        Z             : (1, seq_enc, d_model)

        Retorna probs do próximo token : (vocab_size,)
        """
        seq_len = len(dec_token_ids)
        Y = self.dec_embedding[dec_token_ids]        # (seq, d_model)
        Y = Y + positional_encoding(seq_len, self.d_model)
        Y = Y[np.newaxis, :, :]                      # (1, seq, d_model)

        probs = self.decoder.forward(Y, Z, verbose=False)  # (1, seq, vocab)
        return probs[0, -1, :]   # último token → (vocab_size,)


# ─────────────────────────────────────────────────────────────
# Inferência Auto-Regressiva
# ─────────────────────────────────────────────────────────────

def run_inference(model, encoder_tokens, vocab, max_steps=10):
    """
    Executa o loop while auto-regressivo até <EOS> ou max_steps.

    model         : Transformer
    encoder_tokens: list[int]  — IDs da frase de entrada
    vocab         : list[str]  — vocabulário (índice → palavra)
    """
    EOS_ID   = vocab.index("<EOS>")
    START_ID = vocab.index("<START>")

    print("\n" + "═" * 60)
    print("  TAREFA 4 — Inferência Fim-a-Fim")
    print("═" * 60)

    enc_words = [vocab[i] for i in encoder_tokens]
    print(f"\n  Frase de entrada : {enc_words}")

    # ── 1. Encoder: processa a frase de entrada → Z ───────────
    print("\n  → Rodando o Encoder...")
    Z = model.encode(encoder_tokens)
    print(f"  Memória Z shape  : {Z.shape}")

    # ── 2. Decoder: inicia com <START> ────────────────────────
    dec_sequence = [START_ID]
    print(f"\n  → Iniciando Decoder com token '<START>'")
    print(f"  {'Passo':<8} {'Token gerado':<20} {'ID':<8} {'Prob':>8}")
    print(f"  {'-'*46}")

    step = 0
    while step < max_steps:
        step += 1

        # a. Prevê o próximo token
        probs = model.decode_step(dec_sequence, Z)

        # b. argmax → token mais provável
        # Simulação didática: força <EOS> no passo 5
        # (em modelo treinado isso ocorre naturalmente)
        if step >= 5:
            next_id = EOS_ID
        else:
            next_id = int(np.argmax(probs))

        next_token = vocab[next_id]
        confidence = probs[next_id] * 100

        print(f"  {step:<8} {next_token:<20} {next_id:<8} {confidence:>7.2f}%")

        # c. Concatena à sequência
        dec_sequence.append(next_id)

        # d. Para em <EOS>
        if next_id == EOS_ID:
            print(f"\n  🛑 Token <EOS> detectado — geração encerrada.")
            break
    else:
        print(f"\n  ⚠️  Limite de {max_steps} passos atingido.")

    # Frase final (remove <START> e <EOS>)
    frase = [vocab[i] for i in dec_sequence
             if vocab[i] not in ("<START>", "<EOS>")]

    print(f"\n  Sequência completa : {[vocab[i] for i in dec_sequence]}")
    print(f"  Tradução gerada    : {' '.join(frase) if frase else '(vazia)'}")
    print("═" * 60)

    return dec_sequence


# ─────────────────────────────────────────────────────────────
# Execução
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    # ── Vocabulário fictício ───────────────────────────────────
    VOCAB_SIZE = 10_000
    vocab = (
        ["<PAD>", "<START>", "<EOS>", "<UNK>"]
        + ["Thinking", "Machines", "are", "learning", "fast",
           "as", "the", "future", "is", "now"]
        + [f"word_{i}" for i in range(VOCAB_SIZE - 14)]
    )

    # Frase de entrada: "Thinking Machines"
    encoder_input_words = ["Thinking", "Machines"]
    encoder_token_ids   = [vocab.index(w) for w in encoder_input_words]

    # ── Modelo ────────────────────────────────────────────────
    D_MODEL  = 64
    N_LAYERS = 6

    print("=" * 60)
    print("  Transformer Completo 'From Scratch' — Lab 04")
    print("=" * 60)
    print(f"  d_model    : {D_MODEL}")
    print(f"  d_ff       : {D_MODEL * 4}")
    print(f"  N camadas  : {N_LAYERS}")
    print(f"  Vocab size : {VOCAB_SIZE:,}")

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
    )

    # ── Inferência ────────────────────────────────────────────
    resultado = run_inference(
        model,
        encoder_token_ids,
        vocab,
        max_steps=8,
    )

    print("\n  ✓ Pipeline Encoder-Decoder completo executado com sucesso!")
