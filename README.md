# Laboratório 4 — O Transformer Completo "From Scratch"

**Disciplina:** Tópicos em Inteligência Artificial – 2026.1  
**Professor:** Prof. Dimmy Magalhães  
**Instituição:** iCEV - Instituto de Ensino Superior  

---

## Descrição

Integração completa da arquitetura **Encoder-Decoder** do Transformer,
reunindo os blocos construídos nos Labs 01, 02 e 03 em um único pipeline
funcional de tradução fim-a-fim (*toy sequence*).  
Implementado **sem** PyTorch, TensorFlow ou Keras — apenas `Python 3.x` e `numpy`.

---

## Estrutura do Repositório

```
lab4_transformer/
│
├── main.py                  # Pipeline completo — executa as 4 tarefas
├── task1_building_blocks.py # Tarefa 1: Attention, FFN, Add & Norm
├── task2_encoder.py         # Tarefa 2: EncoderBlock + EncoderStack (N=6)
├── task3_decoder.py         # Tarefa 3: DecoderBlock + DecoderStack + Softmax
├── task4_inference.py       # Tarefa 4: Transformer completo + loop auto-regressivo
├── requirements.txt         # Dependências do projeto
└── README.md                # Este arquivo
```

---

## Arquitetura Implementada

```
  Entrada: "Thinking Machines"
        ↓
  [Embedding + Positional Encoding]
        ↓
  ┌─────────────────────────────────────┐
  │         ENCODER  ×  6              │
  │  Self-Attention (sem máscara)       │
  │  Add & Norm                         │
  │  FFN                                │
  │  Add & Norm                         │
  └──────────────┬──────────────────────┘
                 │  Z  (memória rica)
                 ▼
  ┌─────────────────────────────────────┐
  │         DECODER  ×  6              │
  │  Masked Self-Attention (-inf mask)  │
  │  Add & Norm                         │
  │  Cross-Attention (Q=dec, K/V=enc)   │
  │  Add & Norm                         │
  │  FFN                                │
  │  Add & Norm                         │
  └──────────────┬──────────────────────┘
                 ↓
  [Projeção Linear → vocab_size]
                 ↓
  [Softmax → distribuição de probs]
                 ↓
  Loop auto-regressivo até <EOS>
```

### Componentes por tarefa

| Tarefa | Arquivo | Componentes |
|--------|---------|-------------|
| 1 | `task1_building_blocks.py` | `scaled_dot_product_attention`, `FeedForwardNetwork`, `add_and_norm`, `layer_norm`, `softmax`, `create_causal_mask` |
| 2 | `task2_encoder.py` | `EncoderBlock`, `EncoderStack` (N=6) |
| 3 | `task3_decoder.py` | `DecoderBlock`, `DecoderStack` + projeção Linear + Softmax |
| 4 | `task4_inference.py` | `Transformer`, `positional_encoding`, `run_inference` (loop while) |

### Hiperparâmetros

| Parâmetro | Paper original | Este projeto |
|-----------|---------------|--------------|
| `d_model` | 512 | **64** (CPU) |
| `d_ff`    | 2048 | **256** (4 × d_model) |
| `N` (camadas) | 6 | **6** |
| `vocab_size` | — | **10.000** |

---

## Pré-requisitos

- Python 3.8 ou superior

---

## Como rodar

### 1. Clone o repositório

```bash
git clone https://github.com/<seu-usuario>/lab4-transformer-completo.git
cd lab4-transformer-completo
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Execute o pipeline completo

```bash
python main.py
```

### 4. Execute as tarefas individualmente (opcional)

```bash
python task1_building_blocks.py   # Tarefa 1 isolada
python task2_encoder.py           # Tarefa 2 isolada
python task3_decoder.py           # Tarefa 3 isolada
python task4_inference.py         # Tarefa 4 isolada
```

---

## Saída Esperada

```
============================================================
  Lab 04 — Transformer Completo 'From Scratch'
============================================================
  TAREFA 1 — Blocos Base...    ✓
  TAREFA 2 — Encoder Stack...  ✓ shape preservado: (1, 6, 64)
  TAREFA 3 — Decoder Stack...  ✓ probs shape: (1, 4, 10000)

  TAREFA 4 — Inferência Fim-a-Fim
  Frase de entrada : ['Thinking', 'Machines']
  Passo 1  word_XXXX   ...
  Passo 5  <EOS>       ...
   Token <EOS> detectado — geração encerrada.

  ✓ Lab 04 — Pipeline completo executado com sucesso!
```

---

## Nota de Integridade Acadêmica

Partes geradas/complementadas com IA (Claude).  
O uso foi restrito a brainstorming de sintaxe NumPy e validação das equações
matemáticas do paper, conforme permitido pelo contrato pedagógico da disciplina.
A lógica de integração dos módulos e o entendimento do fluxo de tensores
foram desenvolvidos e documentados pelo aluno.

---

## Referência

Vaswani, A., Shazeer, N., Parmar, N., et al. (2017).  
*Attention Is All You Need*. NeurIPS 2017.  
[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
