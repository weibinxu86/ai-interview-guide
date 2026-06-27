# 🏗️ Transformer Architecture & Attention Mechanism — Interview Questions (English)

> **Difficulty:** ⭐⭐⭐⭐
> **Updated:** 2026-06-27
> **Focus:** Transformer, Self-Attention, BERT, GPT, Positional Encoding

## 📋 Contents

1. [Transformer Fundamentals](#1-transformer-fundamentals)
2. [Attention Mechanism](#2-attention-mechanism)
3. [BERT & GPT](#3-bert--gpt)
4. [Optimization Techniques](#4-optimization-techniques)
5. [Quick Reference Cards](#5-quick-reference-cards)

---

## 1. Transformer Fundamentals

### Q1: What is Transformer? Why do we need Transformer?

**Transformer = Sequence-to-sequence model based on self-attention mechanism**

#### Why do we need Transformer?

**Problems with RNN/LSTM:**

| Problem | Description | Impact |
|---------|-------------|--------|
| **Sequential Computation** | Must process tokens one by one | Slow training, no parallelization |
| **Long-term Dependencies** | Vanishing/exploding gradients | Difficult to capture distant relationships |
| **Information Bottleneck** | All information compressed into hidden state | Information loss |

**Transformer Advantages:**

```
┌─────────────────────────────────────────────────────────┐
│                 Transformer Architecture                │
└─────────────────────────────────────────────────────────┘

Input Sequence → Embedding + Positional Encoding
     ↓
Encoder (N × 6 layers)
├── Multi-Head Self-Attention
├── Add & Norm
├── Feed-Forward Network
└── Add & Norm
     ↓
Decoder (N × 6 layers)
├── Masked Multi-Head Self-Attention
├── Add & Norm
├── Encoder-Decoder Cross-Attention
├── Add & Norm
├── Feed-Forward Network
└── Add & Norm
     ↓
Linear + Softmax → Output probability distribution
```

**Core Innovations:**

1. **Self-Attention (Self-Attention)**
   - Each token can directly "see" all other tokens
   - Complexity: O(n²), but can be parallelized

2. **Multi-Head Attention (Multi-Head Attention)**
   - Multiple attention heads capture relationships from different dimensions
   - 8 or 16 heads

3. **Positional Encoding (Positional Encoding)**
   - Inject sequence position information
   - Using sin/cos functions

4. **Residual Connection + Layer Norm**
   - Solve vanishing gradients
   - Stabilize training

**Performance Comparison (Machine Translation Task):**

| Model | BLEU | Training Time | Parameters |
|-------|------|---------------|-----------|
| LSTM | 25.3 | 10 days | 200M |
| Transformer Base | 27.3 | 12 hours | 65M |
| Transformer Big | 28.4 | 3.5 days | 213M |

**Interview Answer:**
> "Transformer replaced RNN's sequential computation with self-attention mechanism. Each token can directly attend to all other tokens, enabling parallel computation and achieving 10-100x faster training. Although complexity is O(n²), the parallelization and training speed make it highly efficient."

---

### Q2: What's the difference between Encoder and Decoder in Transformer?

**Core Difference: Encoder is bidirectional, Decoder is unidirectional**

#### Encoder (Encoder):

```python
# Encoder structure (repeated N times, typically N=6)
for layer in range(N):
    # 1. Multi-Head Self-Attention
    # Can see the entire input sequence (bidirectional)
    attn_output = MultiHeadAttention(
        Q=x, K=x, V=x  # Query, Key, Value all from input
    )
    x = LayerNorm(x + attn_output)  # Residual + Normalization

    # 2. Feed-Forward Network
    ffn_output = FeedForward(x)
    x = LayerNorm(x + ffn_output)
```

**Characteristics:**
- ✅ Bidirectional attention (can see context before and after)
- ✅ Process all tokens in parallel
- ✅ Output: encoded representations (context vectors)

#### Decoder (Decoder):

```python
# Decoder structure (repeated N times, typically N=6)
for layer in range(N):
    # 1. Masked Multi-Head Self-Attention
    # Can only see current and previous tokens (unidirectional)
    masked_attn = MaskedMultiHeadAttention(
        Q=y, K=y, V=y,  # From target sequence
        mask=causal_mask  # Upper triangular mask
    )
    y = LayerNorm(y + masked_attn)

    # 2. Encoder-Decoder Cross-Attention
    # Query from Decoder, Key and Value from Encoder output
    cross_attn = MultiHeadAttention(
        Q=y,  # From Decoder
        K=encoder_output,  # From Encoder
        V=encoder_output   # From Encoder
    )
    y = LayerNorm(y + cross_attn)

    # 3. Feed-Forward Network
    ffn_output = FeedForward(y)
    y = LayerNorm(y + ffn_output)
```

**Characteristics:**
- ⚠️ Unidirectional attention (Masked, can only see previous tokens)
- ✅ Includes Cross-Attention (connects Encoder and Decoder)
- ⚠️ Autoregressive generation (generate tokens one by one)

**Key Differences:**

| Dimension | Encoder | Decoder |
|-----------|---------|---------|
| **Self-Attention** | Bidirectional (no mask) | Unidirectional (with mask) |
| **Cross-Attention** | ❌ None | ✅ Yes (connects to Encoder) |
| **Input** | Source sequence (e.g., English) | Target sequence (e.g., Chinese) |
| **Output** | Encoded representations | Generated sequence |
| **Application** | BERT (Encoder only) | GPT (Decoder only) |

**Masking Mechanism Explained:**

```
# Encoder: No mask, all tokens can see each other
Input:  [I, love, AI]
Attention Matrix:
      I   love  AI
I    ✓    ✓    ✓
love ✓    ✓    ✓
AI   ✓    ✓    ✓

# Decoder: Causal Mask, can only see current and previous
Input:  [I, love, AI]
Attention Matrix:
       I  love  AI
I      ✓   ✗    ✗
love   ✓   ✓    ✗
AI     ✓   ✓    ✓
```

**Interview Answer:**
> "Encoder uses bidirectional Self-Attention to understand the input, Decoder uses unidirectional Masked Attention to generate outputs. Decoder also has a Cross-Attention layer, allowing each generated token to attend to all Encoder outputs. BERT uses only Encoder, while GPT uses only Decoder."

---

## 2. Attention Mechanism

### Q3: How is Self-Attention calculated?

**Self-Attention = Let each element in the sequence attend to all other elements**

#### Mathematical Formula:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

#### Detailed Steps:

**Step 1: Generate Q, K, V:**

```python
# Input: X (batch_size, seq_len, d_model)
# Example: (1, 5, 512)

# Generate Q, K, V through linear transformation
Q = X @ W_Q  # (batch_size, seq_len, d_k)
K = X @ W_K  # (batch_size, seq_len, d_k)
V = X @ W_V  # (batch_size, seq_len, d_v)

# W_Q, W_K, W_V are learnable weight matrices
# d_k = d_v = d_model / num_heads (typically 64)
```

**Step 2: Calculate Attention Scores:**

```python
# Compute similarity using dot product
scores = Q @ K.T  # (batch_size, seq_len, seq_len)
# Example: (1, 5, 5)

# Scaling (prevent vanishing gradients)
scores = scores / math.sqrt(d_k)

# Apply mask (optional, used in Decoder)
if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
```

**Step 3: Softmax Normalization:**

```python
# Apply softmax to each row to get attention weights
attention_weights = softmax(scores, dim=-1)
# (batch_size, seq_len, seq_len)

# Weights sum to 1
assert attention_weights.sum(dim=-1) == 1.0
```

**Step 4: Weighted Sum:**

```python
# Weight V using attention weights
output = attention_weights @ V
# (batch_size, seq_len, d_v)
```

#### Complete Example (sequence length=3):

```python
# Input
X = [[1.0, 0.5],  # token 1
     [0.8, 1.0],  # token 2
     [0.5, 0.9]]  # token 3

# Assume W_Q = W_K = W_V = I (identity matrix)
Q = K = V = X

# 1. Calculate similarity matrix
scores = Q @ K.T
# [[1.25, 1.3, 0.95],
#  [1.3,  1.64, 1.3],
#  [0.95, 1.3, 1.06]]

# 2. Scaling (assuming d_k=2)
scores = scores / sqrt(2)
# [[0.88, 0.92, 0.67],
#  [0.92, 1.16, 0.92],
#  [0.67, 0.92, 0.75]]

# 3. Softmax
attention_weights = softmax(scores, dim=-1)
# [[0.32, 0.35, 0.33],  # token 1's attention to 3 tokens
#  [0.28, 0.44, 0.28],  # token 2's attention to 3 tokens
#  [0.29, 0.37, 0.34]]  # token 3's attention to 3 tokens

# 4. Weighted sum
output = attention_weights @ V
# [[0.77, 0.80],  # token 1's output
#  [0.76, 0.87],  # token 2's output
#  [0.74, 0.82]]  # token 3's output
```

#### Why Scale by √d_k?

```
Problem:
  When d_k is large (e.g., 512), QK^T values become very large
  → Softmax gradient approaches 0 (saturation)
  → Vanishing gradient

Solution:
  Divide by √d_k, scale variance to 1
  → Keep Softmax input in reasonable range
  → Stable gradient
```

**Interview Answer:**
> "Self-Attention computes through Q, K, V matrices. First calculate similarity using QK^T, divide by √d_k to prevent vanishing gradients, then softmax to normalize and get weights, finally weighted sum of V to get output. The core innovation is allowing each token to directly attend to all other tokens in parallel."

---

### Q4: What is Multi-Head Attention? Why use multiple heads?

**Multi-Head Attention = Multiple Self-Attention in parallel, capturing different types of relationships**

#### Why Multiple Heads?

**Limitations of Single Head:**

```
Single attention head can only learn one pattern
Example:
  "I love eating apples"

Single head might only focus on:
  Syntactic relationship: "I" ← "love" (subject-verb)

But misses:
  Semantic relationship: "eating" ← "apples" (verb-object)
  Coreference: "I" → "I" (self-reference)
```

**Advantages of Multiple Heads:**

```
8 heads can learn different patterns:
  Head 1: Syntactic relationships (subject-verb-object)
  Head 2: Semantic relationships (entity-action)
  Head 3: Positional relationships (adjacent words)
  Head 4: Long-range dependencies (sentence start-end)
  ...
  Head 8: Other patterns
```

#### Architecture:

```python
class MultiHeadAttention:
    def __init__(self, d_model=512, num_heads=8):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 64

        # Each head has independent Q, K, V weights
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Output projection
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 1. Linear transformation and split heads
        # (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k)

        # Transpose for parallel computation of multiple heads
        # (batch, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 2. Calculate attention independently for each head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)

        # 3. Merge multiple heads
        # (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.num_heads * self.d_k)

        # 4. Output projection
        output = self.W_O(output)

        return output
```

#### Visualization Example (8 heads):

```
Input: "The cat sat on the mat"

Head 1 (Subject-Verb):
  cat → sat (0.9)

Head 2 (Verb-Object):
  sat → mat (0.8)

Head 3 (Modification):
  the → cat (0.7)
  the → mat (0.6)

Head 4 (Positional):
  on → the (0.9)

...

Final Output = Concat(Head1, Head2, ..., Head8) @ W_O
```

#### Parameter Comparison:

| Approach | Parameters | Expressiveness |
|----------|-----------|-----------------|
| Single Head (d_model=512) | 512² × 3 = 786K | Low |
| 8 Heads (d_k=64) | 512² × 3 + 512² = 1.05M | **High** |

#### Experimental Evidence (BLEU score, Machine Translation):

| Heads | BLEU | Notes |
|------|------|-------|
| 1 | 25.8 | Single head |
| 2 | 26.4 | +0.6 |
| 4 | 27.1 | +1.3 |
| 8 | **27.3** | **Best** |
| 16 | 27.2 | Too many, performance drops |

**Interview Answer:**
> "Multi-Head Attention allows the model to learn multiple attention patterns simultaneously. 8 heads can focus on different aspects like syntax, semantics, and positions respectively. Although parameters increase slightly, expressiveness increases significantly. Experiments show 8 heads is optimal for most tasks."

---

### Q5: What is Positional Encoding? Why do we need it?

**Positional Encoding = Inject position information into each token**

#### Why Do We Need It?

```
Problem:
  Self-Attention is permutation-invariant

Example:
  "I love you" and "you love me"
  Without positional encoding, Attention output might be similar
  → Cannot distinguish order

Solution:
  Add position information, let model know each token's position
```

#### Transformer's Positional Encoding (Sinusoidal):

**Formula:**

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
  pos: position (0, 1, 2, ...)
  i:   dimension index (0 to d_model/2)
  d_model: model dimension (e.g., 512)
```

**Implementation:**

```python
def positional_encoding(max_len, d_model):
    # Create position matrix
    position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)

    # Create dimension matrix
    div_term = torch.exp(
        torch.arange(0, d_model, 2) *
        -(math.log(10000.0) / d_model)
    )  # (d_model/2,)

    # Initialize positional encoding
    pe = torch.zeros(max_len, d_model)

    # Even dimensions use sin
    pe[:, 0::2] = torch.sin(position * div_term)

    # Odd dimensions use cos
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

# Usage
pe = positional_encoding(max_len=100, d_model=512)
# pe.shape: (100, 512)

# Add to input
x = embedding(input_ids) + pe[: input_ids.size(1)]
```

#### Why Use Sin/Cos?

**Advantages:**

1. **Deterministic**: Fixed positions, no need to learn
2. **Extrapolation**: Can handle sequences longer than training data
3. **Periodicity**: Regular relative relationships between positions

**Relative Position Relationship:**

```python
# PE(pos + k) can be expressed as linear combination of PE(pos)
# This allows model to learn relative positions

Example:
  Distance between PE(5) and PE(10)
  = Distance between PE(1) and PE(6)
  (Both relative distance is 5)
```

#### Visualization (First 10 positions, first 64 dimensions):

```
Position  Dim0   Dim1   Dim2   Dim3  ...
0         0.00   1.00   0.00   1.00
1         0.84   0.54   0.01   1.00
2         0.91  -0.42   0.02   1.00
3         0.14  -0.99   0.03   1.00
...
```

#### Alternative Positional Encoding Schemes:

| Scheme | Pros | Cons | Application |
|--------|------|------|-------------|
| **Sinusoidal** | Deterministic, good extrapolation | Fixed, not learnable | Transformer |
| **Learned** | Learnable, flexible | Cannot extrapolate | BERT |
| **RoPE** | Relative position, rotation-based | Complex to implement | LLaMA |
| **ALiBi** | Attention bias | Needs retraining | BLOOM |

#### Modern Improvement: RoPE (Rotary Position Embedding):

```python
# Used by LLaMA/GPT-NeoX
# Core idea: Rotate Q and K during attention computation

def apply_rotary_emb(q, k, cos, sin):
    # Rotate Q and K
    q_rot = rotate_half(q)
    k_rot = rotate_half(k)

    q = q * cos + q_rot * sin
    k = k * cos + k_rot * sin

    return q, k

# Advantages:
#   - Relative position encoding
#   - Better extrapolation
#   - Better performance on long texts
```

**Interview Answer:**
> "Transformer injects position information using sin/cos functions. Advantages include deterministic nature, ability to extrapolate to longer sequences, and expressing relative positions. Modern models like LLaMA use RoPE improvement, rotating Q and K during attention computation, achieving better performance on long texts."

---

## 3. BERT & GPT

### Q6: What's the difference between BERT and GPT?

**Core Difference: BERT is bidirectional understanding, GPT is unidirectional generation**

#### Architecture Comparison:

| Dimension | BERT | GPT |
|-----------|------|-----|
| **Architecture** | Encoder only | Decoder only |
| **Attention** | Bidirectional (no mask) | Unidirectional (causal mask) |
| **Training Objective** | MLM + NSP | Autoregressive language model |
| **Task Type** | Understanding (classification, NER) | Generation (text generation) |
| **Parameters** | 110M-340M | 117M-175B |

#### BERT (Bidirectional Encoder Representations from Transformers):

```
┌─────────────────────────────────────────────────────────┐
│                   BERT Architecture                     │
└─────────────────────────────────────────────────────────┘

Input: "The [MASK] sat on the mat"
   ↓
Embedding + Positional Encoding
   ↓
Transformer Encoder (12/24 layers)
├── Multi-Head Self-Attention (bidirectional)
└── Feed-Forward Network
   ↓
Output: Context representation for each token
   ↓
Task Head (classification/NER/...)
```

**Training Objective 1: MLM (Masked Language Model)**

```python
# Randomly mask 15% of tokens
Original: "The cat sat on the mat"
Masked:   "The [MASK] sat on the [MASK]"

# Predict masked words
Loss = CrossEntropy(predicted, ["cat", "mat"])

# 15% masking strategy:
#   80%: Replace with [MASK]
#   10%: Replace with random word
#   10%: Keep unchanged
```

**Training Objective 2: NSP (Next Sentence Prediction)**

```python
# Determine if two sentences are consecutive
Input:
  A: "The cat sat on the mat."
  B: "It was very comfortable."
  Label: IsNext (1) or NotNext (0)

Loss = BCELoss(predicted, Label)
```

#### GPT (Generative Pre-trained Transformer):

```
┌─────────────────────────────────────────────────────────┐
│                   GPT Architecture                      │
└─────────────────────────────────────────────────────────┘

Input: "The cat sat on"
   ↓
Embedding + Positional Encoding
   ↓
Transformer Decoder (12/96 layers)
├── Masked Multi-Head Self-Attention (unidirectional)
└── Feed-Forward Network
   ↓
Output: Probability distribution of next token
   ↓
Prediction: "the" (probability 0.8)
```

**Training Objective: Autoregressive Language Model**

```python
# Predict next word
Input: "The cat sat"
Target: "on"

# During training, compute loss for all positions in parallel
Loss = Σ CrossEntropy(predicted[i], target[i])

# During inference, generate autoregressively
generated = []
for i in range(max_len):
    next_token = model.predict(generated)
    generated.append(next_token)
```

#### Capability Comparison:

| Task | BERT | GPT |
|------|------|-----|
| **Text Classification** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Named Entity Recognition** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Question Answering** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Text Generation** | ⭐ | ⭐⭐⭐⭐⭐ |
| **Summarization** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Dialogue** | ⭐ | ⭐⭐⭐⭐⭐ |

#### Practical Applications:

**BERT Excels At:**

```python
# 1. Classification
"This movie is great!" → Positive (0.95)

# 2. NER (Named Entity Recognition)
"Apple was founded by Steve Jobs"
→ [Apple: ORG], [Steve Jobs: PER]

# 3. Question Answering
Context: "Paris is the capital of France."
Question: "What is the capital of France?"
Answer: "Paris" (span: [0, 5])
```

**GPT Excels At:**

```python
# 1. Text Generation
Prompt: "Once upon a time"
Output: "there was a brave knight..."

# 2. Dialogue
User: "How are you?"
GPT: "I'm doing well, thank you!"

# 3. Code Generation
Prompt: "Write a Python function to sort a list"
Output: "def sort_list(arr): return sorted(arr)"
```

#### Why BERT Uses Bidirectional, GPT Uses Unidirectional?

```
BERT:
  Goal is language understanding
  Bidirectional can see complete context
  Example: "bank" (financial vs river) needs context

GPT:
  Goal is language generation
  Must be unidirectional, otherwise "cheating"
  During generation, can only see generated part
```

**Interview Answer:**
> "BERT is an understanding model using bidirectional Encoder + MLM training, excels at classification and NER. GPT is a generation model using unidirectional Decoder + autoregressive training, excels at text generation and dialogue. BERT sees complete context on both sides, while GPT only sees past context to avoid 'cheating' during generation."

---

## 4. Optimization Techniques

### Q7: What optimization techniques are used during Transformer training?

**Core Optimization Directions:**

1. **Learning Rate Scheduling**
2. **Regularization**
3. **Training Stability**
4. **Computational Efficiency**

#### 1. Learning Rate Scheduling (Warmup + Decay):

```python
# Warmup phase: linearly increase learning rate
def get_lr(step, d_model, warmup_steps=4000):
    # Transformer paper's approach
    lr = d_model ** (-0.5) * min(
        step ** (-0.5),
        step * warmup_steps ** (-1.5)
    )
    return lr

# Why Warmup?
# - Initially, weights are random, gradients unstable
# - Small learning rate lets model "warm up"
# - Then gradually increase to peak
# - Finally gradually decay

# Typical curve:
#   0-4K steps: linear increase 0 → peak_lr
#   4K+ steps: decay by 1/√step
```

#### 2. Label Smoothing:

```python
# Problem: Hard labels easily overfit
hard_label = [0, 0, 0, 1, 0]  # one-hot

# Solution: Label Smoothing
smooth_label = [0.02, 0.02, 0.02, 0.92, 0.02]
# True class: 0.92 (1 - smoothing)
# Other classes: 0.02 (smoothing / (n_classes - 1))

class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, pred, target):
        # pred: (batch, n_classes)
        # target: (batch,) class indices

        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.n_classes - 1)

        # Construct smooth label
        smooth_label = torch.full_like(pred, smooth_value)
        smooth_label.scatter_(1, target.unsqueeze(1), confidence)

        # KL divergence loss
        loss = -torch.sum(smooth_label * torch.log_softmax(pred, dim=1), dim=1)
        return loss.mean()

# Effect: Improves generalization, BLEU +0.2
```

#### 3. Dropout Strategy:

```python
class TransformerLayer(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        # 1. Attention Dropout
        attn_output = self.attention(x)
        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)

        # 2. Residual Dropout
        x = x + attn_output

        # 3. FFN Dropout
        ffn_output = self.ffn(x)
        ffn_output = F.dropout(ffn_output, p=self.dropout, training=self.training)

        x = x + ffn_output

        return x

# Typical configuration:
#   Attention Dropout: 0.1
#   Residual Dropout: 0.1
#   FFN Dropout: 0.1
#   Embedding Dropout: 0.1
```

#### 4. Gradient Clipping:

```python
# Prevent gradient explosion
max_grad_norm = 1.0

for batch in dataloader:
    loss = model(batch)
    loss.backward()

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm
    )

    optimizer.step()
    optimizer.zero_grad()
```

#### 5. Mixed Precision Training:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # FP16 forward pass
    with autocast():
        loss = model(batch)

    # Scale loss, FP16 backward pass
    scaler.scale(loss).backward()

    # Update weights (FP32)
    scaler.step(optimizer)
    scaler.update()

# Advantages:
#   - 2-3x speed improvement
#   - 50% memory savings
#   - Accuracy loss < 0.1%
```

#### 6. Batch Size Optimization:

```python
# Problem: Limited GPU memory, batch_size constrained

# Solution: Gradient Accumulation
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Effect:
#   Original: batch_size=32, update each step
#   Now: batch_size=32×4=128, update every 4 steps
#   Memory unchanged, better results
```

#### 7. Parallel Training:

```python
# Data Parallel
model = nn.DataParallel(model)

# Distributed Data Parallel (recommended)
model = nn.parallel.DistributedDataParallel(model)

# Model Parallel (for very large models)
# Place different layers on different GPUs
class ModelParallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(*layers[:6]).to('cuda:0')
        self.decoder = nn.Sequential(*layers[6:]).to('cuda:1')

    def forward(self, x):
        x = self.encoder(x.to('cuda:0'))
        x = self.decoder(x.to('cuda:1'))
        return x
```

#### Typical Configuration (Transformer Base):

| Parameter | Value |
|-----------|-------|
| d_model | 512 |
| n_heads | 8 |
| d_ff | 2048 |
| n_layers | 6 |
| dropout | 0.1 |
| warmup_steps | 4000 |
| label_smoothing | 0.1 |
| max_grad_norm | 1.0 |
| batch_size | 25K tokens |
| optimizer | Adam(β1=0.9, β2=0.98, ε=1e-9) |

**Interview Answer:**
> "Key Transformer training optimizations include: warmup learning rate scheduling (ramp up then decay), label smoothing to prevent overfitting, multi-layer dropout for regularization, gradient clipping to prevent explosion. Engineered techniques include mixed precision training for 2-3x speedup, gradient accumulation for larger effective batch sizes, and distributed training across multiple GPUs."

---

### Q8: What's the detailed calculation process for Q, K, V matrices?

**Q, K, V = Query(Query), Key(Key), Value(Value)**

#### Generation Process:

**Step 1: Input Embedding**

```python
# Input sequence: "I love AI"
input_ids = [101, 234, 567]  # token IDs
embeddings = embedding_layer(input_ids)  # shape: (3, 512)
# Each token → 512-dimensional vector

# Add positional encoding
position_encodings = get_position_encoding(3, 512)
input_repr = embeddings + position_encodings  # shape: (3, 512)
```

**Step 2: Linear Transformation to Generate Q, K, V**

```python
# 3 learnable weight matrices
W_Q = nn.Linear(512, 512)  # Query weight
W_K = nn.Linear(512, 512)  # Key weight
W_V = nn.Linear(512, 512)  # Value weight

# Generate Q, K, V
Q = W_Q(input_repr)  # shape: (3, 512)
K = W_K(input_repr)  # shape: (3, 512)
V = W_V(input_repr)  # shape: (3, 512)
```

#### Why 3 Matrices?

- **Q (Query)**: "What information do I want to find?"
- **K (Key)**: "What information can I provide?"
- **V (Value)**: "What information do I contain?"

#### Search Engine Analogy:

```
Q = User search query "Python tutorial"
K = Document titles ["Python intro", "Java tutorial", "Python advanced"]
V = Document content [actual Python tutorial texts]

Steps:
1. Calculate similarity between Q and each K → attention scores
2. Weight V using scores → final output
```

#### Multi-Head Attention Detailed Calculation:

**Step 1: Split into Multiple Heads**

```python
num_heads = 8
d_k = 512 // 8 = 64

# Reshape Q, K, V
# (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
Q_split = Q.view(batch_size, seq_len, num_heads, d_k)
K_split = K.view(batch_size, seq_len, num_heads, d_k)
V_split = V.view(batch_size, seq_len, num_heads, d_k)

# Transpose to parallel compute
# (batch, num_heads, seq_len, d_k)
Q_split = Q_split.transpose(1, 2)
K_split = K_split.transpose(1, 2)
V_split = V_split.transpose(1, 2)
```

**Step 2: Compute Attention for Each Head**

```python
# For each head, compute attention
scores = torch.matmul(Q_split, K_split.transpose(-2, -1))
# (batch, num_heads, seq_len, seq_len)

scores = scores / math.sqrt(d_k)  # Scale

attention_weights = F.softmax(scores, dim=-1)
# (batch, num_heads, seq_len, seq_len)

head_output = torch.matmul(attention_weights, V_split)
# (batch, num_heads, seq_len, d_k)
```

**Step 3: Concatenate Heads**

```python
# Transpose back
# (batch, seq_len, num_heads, d_k)
head_output = head_output.transpose(1, 2).contiguous()

# Concatenate all heads
# (batch, seq_len, num_heads * d_k) = (batch, seq_len, 512)
multi_head_output = head_output.view(batch_size, seq_len, d_model)
```

**Step 4: Output Projection**

```python
# Final linear transformation
W_O = nn.Linear(512, 512)
final_output = W_O(multi_head_output)
# (batch, seq_len, 512)
```

#### Complete PyTorch Example:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear transformation
        Q = self.W_Q(Q)  # (batch, seq_len, d_model)
        K = self.W_K(K)
        V = self.W_V(V)
        
        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # (batch, num_heads, seq_len, d_k)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        
        # Weighted sum
        output = torch.matmul(attention, V)
        # (batch, num_heads, seq_len, d_k)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        
        # Output projection
        output = self.W_O(output)
        
        return output

# Usage example
batch_size, seq_len, d_model = 2, 10, 512
input_tensor = torch.randn(batch_size, seq_len, d_model)

attention = MultiHeadAttention(d_model=d_model, num_heads=8)
output = attention(input_tensor, input_tensor, input_tensor)
print(output.shape)  # (2, 10, 512)
```

**Interview Answer:**
> "Q, K, V are generated through linear transformations of input representations. In multi-head attention: (1) generate Q, K, V from input, (2) split into 8 heads with d_k=64, (3) compute attention independently for each head, (4) concatenate head outputs, (5) apply final linear projection. This allows the model to simultaneously capture different relationship types."

---

## 5. Quick Reference Cards

### Key Formulas

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Multi-Head(Q, K, V) = Concat(head_1, ..., head_h)W^O
  where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Transformer Dimensions

| Component | Input | Output | Notes |
|-----------|-------|--------|-------|
| Embedding | (batch, seq_len) | (batch, seq_len, 512) | Token → vectors |
| Positional Encoding | (max_len, 512) | (max_len, 512) | Added to embedding |
| Q/K/V Projection | (batch, seq_len, 512) | (batch, seq_len, 512) | Per head: 64 dims |
| Attention | (batch, 8, seq_len, 64) | (batch, 8, seq_len, 64) | 8 heads |
| Concatenate | 8×(batch, seq_len, 64) | (batch, seq_len, 512) | Merge heads |
| Output Projection | (batch, seq_len, 512) | (batch, seq_len, 512) | Final output |

### Comparison Table

| Aspect | Transformer | RNN/LSTM | CNN |
|--------|-------------|----------|-----|
| **Parallelization** | ✅ Full | ❌ No | ✅ Full |
| **Long-range Dependencies** | ✅ Direct | ❌ Gradient issues | ⚠️ Limited window |
| **Computational Complexity** | O(n²) | O(n) | O(n) |
| **Memory Usage** | High | Medium | Low |
| **Interpretability** | ✅ Attention weights | Medium | Low |
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

---

## Interview Tips

1. **Explain the why, not just the what**
   - Don't just say "use attention," explain why it's better than RNN

2. **Know the dimensions**
   - Be able to trace tensor shapes through the model

3. **Provide examples**
   - Use concrete sentences like "I love AI" when explaining concepts

4. **Discuss trade-offs**
   - O(n²) complexity vs. parallelization benefits

5. **Mention practical considerations**
   - Training tricks, optimization techniques, real-world issues

---
