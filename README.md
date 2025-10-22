# Learn-Bert-fine-tuning
BERT (Bidirectional Encoder Representations from Transformers) = A Transformer Encoder stack that reads a sentence both ways and learns meaning from context.
It converts text → dense numerical vectors (embeddings) using attention and matrix math.

# How text becomes math — Tokenization
Example:
| Token | ID    |
| ----- | ----- |
| [CLS] | 101   |
| BERT  | 14324 |
| loves | 2347  |
| math  | 4562  |
| .     | 119   |
| [SEP] | 102   |

Sentence = [101, 14324, 2347, 4562, 119, 102]

# Token Embeddings — The math of “words”

Each token ID is turned into a vector of 768 numbers (for base BERT).
This happens via an Embedding Matrix E of size 30522 × 768

If token ID = 2347, then row 2347 of E gives a 768-dim vector like: [0.14, -0.26, 0.78, ..., 0.11]

# Transformer Encoder Math (the core of BERT)

Each layer of BERT has two sublayers:

Self-Attention layer
Feed-forward network

There are 12 layers (in BERT-base).

## Self-Attention: "Which words matter to this word?"

BERT computes three matrices per token:

Q (Query)

K (Key)

V (Value)

Each is derived via matrix multiplication:
Q=X*WQ
K=X*WK
V=X*WV​

So, each token now has:

a query vector (what it’s looking for)
a key vector (what it offers)
a value vector (the content)

## Attention Scores

We find how much each token should attend to every other token:
Attention Score=​​Q*KT​/dk^(1/2)

Attention Output = softmax(Q*KT​/dk^(1/2))V

This happens in multiple heads (12 in base BERT), hence the term multi-head attention.
Finally, outputs are concatenated and passed through a small feed-forward neural net and normalization.

## Masked Language Modeling (MLM) — The learning objective

BERT learns by masking 15% of the tokens and predicting them.
Models must predict the masked word using both left and right context (bidirectional).
That’s why it’s called Bidirectional Encoder Representations.

Loss function: Loss = −∑logP(true token | masked context) - what is the probability of getting a close value to the true token, given that we have masked actual words.

## Sentence Pair Task (Next Sentence Prediction)

During pretraining, BERT also sees sentence pairs:
A and B that follow each other (50%)
Random B (50%)

It learns to predict whether B follows A or not.
This helps BERT understand relationships between sentences — crucial for QA and reasoning.

## After 12 layers of attention and transformation:

Each token → 768-dim vector (context-aware)

[CLS] token → represents the whole sentence

You can then:

Use [CLS] for classification (sentiment, intent, etc.)

Use token embeddings for NER, QA, etc.

## What “understanding” means mathematically

BERT doesn’t understand language like humans —

it learns vector geometry where similar meanings lie close in space.

For example:
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
“good” and “excellent” have small cosine distance
“bank” in “river bank” vs “money bank” differ due to context

# Important Hyperparameters

Learning Rate

Typical range: 2e-5 to 5e-5
Lower than training from scratch because BERT is already well-trained
Too high can destroy pre-trained weights

Batch Size

Common values: 8, 16, 32
Limited by GPU memory
Smaller batches may need more epochs

Number of Epochs

Typical range: 2-4 epochs
BERT fine-tunes quickly
More epochs risk overfitting on small datasets

Maximum Sequence Length

Default: 512 tokens
Longer sequences need more memory
Truncate based on your data distribution

# Best Practices

Start Small: Begin with a small subset to test your pipeline

Monitor Overfitting: Use validation set, watch for divergence

Layer Freezing: For very small datasets, freeze early BERT layers

Data Augmentation: Helpful when data is limited

Use Mixed Precision: Enable fp16 training to save memory

Gradient Accumulation: Simulate larger batches if GPU memory is limited
​
