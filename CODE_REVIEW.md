# Project Thai Fake News Detection - Comprehensive Code Review

## 📋 Project Overview

**Project Status:** CS Project 2568 (NOT DONE) 

**Objective:** Develop a fake news detection system for Thai news using deep learning and graph neural networks.

**Key Technology Stack:**
- **Language Model:** WangchanBERTa (Thai-specific BERT)
- **Graph Neural Network:** GAT (Graph Attention Networks) + Temporal embeddings
- **Baselines:** TF-IDF + LR, XLM-RoBERTa + LR, WangchanBERTa + RF
- **Framework:** PyTorch, PyTorch Geometric, FastAPI

---

## 🎯 Project Accomplishments

### 1. **Data Processing Pipeline** ✅
- Successfully loaded Thai news dataset (AFNC_news_dataset_tf-2.csv)
- Implemented comprehensive Thai text normalization:
  - Removed zero-width characters (ZWJ, ZWS, ZWSP, BOM)
  - NFC Unicode normalization
  - Whitespace collapsing and diacritical mark cleanup
- Label encoding: `ข่าวจริง (True) = 0`, `ข่าวปลอม (Fake) = 1`
- Category encoding for news classification

### 2. **Multi-Embedding Architecture** ✅
- **WangchanBERTa embeddings**: 768-dimensional Thai BERT embeddings
- **XLM-RoBERTa embeddings**: Multilingual embeddings for comparison
- **TF-IDF embeddings**: Traditional baseline
- Proper normalization (L2 norm) for cosine similarity

### 3. **Graph Construction & kNN Integration** ✅
- Built k-nearest neighbor (k=10) similarity graphs
- Star graph topology: center node (query) connected to k neighbors
- Edge weighting based on cosine similarity
- Successfully integrated with PyTorch Geometric Data objects

### 4. **Model Implementation** ✅
- **GCN Model** (basic Graph Convolutional Network)
- **GAT Model** (Graph Attention Networks v2) with multi-head attention
- **Baseline Models**: TF-IDF + LR, WangchanBERTa + RF, XLM-RoBERTa + LR
- Training pipeline with train/val/test splits
- Attention weight extraction for explainability

### 5. **Inference System** ✅
- kNN-based prediction function with graph construction
- Explainability features using attention weights
- FastAPI REST API endpoint (`/predict`)
- Proper CUDA/CPU device management

### 6. **Evaluation & Analysis** ✅
- Comprehensive metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrices and classification reports
- Cross-model error analysis
- Hard example identification (samples all 4 models fail on)
- Error type categorization (False Positives vs False Negatives)
- Top keywords analysis for misclassifications

---

## 📊 Model Performance Comparison

Based on the test set evaluation in `test_3.ipynb`:

### Results Table:
```
Model                          Accuracy    Precision   Recall      F1-Score
═════════════════════════════════════════════════════════════════════════
1. GCN (Wangchan+Graph)        ???         ???         ???         ???
2. TF-IDF + LR                 ???         ???         ???         ???
3. WangchanBERTa + RF          ???         ???         ???         ???
4. XLM-RoBERTa + LR            ???         ???         ???         ???
```

**Note:** The actual results are computed in Section 13 (Part 3) of `test_3.ipynb`, but specific numbers weren't extracted from visible output sections.

---

## 🐛 ERRORS & ISSUES IDENTIFIED

### **CRITICAL ERRORS:**

#### 1. **Mean Pooling Implementation Bug (api.py)** 🔴
**Location:** `api.py`, section "1.3 Mean Pooling"
```python
# ❌ WRONG - Classic Mean Pooling Bug
last_hidden = outputs.last_hidden_state  # (1, Seq_Len, 768)
emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Includes [PAD] tokens!
```

**Problem:** The basic mean pooling includes padding tokens, diluting semantic information.

**Solution (correct implementation in api.py):**
```python
# ✅ CORRECT - Attention-masked mean pooling
last_hidden = outputs.last_hidden_state  # (1, Seq_Len, 768)
attn = inputs['attention_mask'].unsqueeze(-1)  # (1, Seq_Len, 1)
summed = (last_hidden * attn).sum(dim=1)       # (1, 768)
denom = attn.sum(dim=1).clamp(min=1)           # (1, 1)
content_emb = (summed / denom).cpu().numpy()[0]  # (768,)
```

---

#### 2. **Temporal Decay Formula Issues** 🔴
**Location:** `test_3.ipynb`, Section 9 (Temporal Graph Building)

```python
alpha = 0.00135  # Half-life of ~512 days
dt_days = abs(int(time_values[i]) - int(time_values[j])) / 86400  # ✅ Correct
temporal_weight = np.exp(-alpha * dt_days)
final_weight = sim * temporal_weight
```

**Problems:**
1. **Time data not extracted**: `time_values` array is never defined/populated from dataset
   - No `publish_date` or timestamp column used
   - Falls back to index positions (meaningless temporal ordering)

2. **Decay constant arbitrary**: α=0.00135 vs actual data distribution not validated
   - No ablation study comparing different α values
   - No analysis of actual timestamp distributions

3. **Multiplicative combination suboptimal**: `sim × temporal_weight` treats similarity and temporal distance as independent
   - May not reflect semantic + temporal decay relationship properly
   - No justification for multiplicative vs additive vs other combinations

---

#### 3. **GAT Implementation Mismatch** 🔴
**Location:** `test_3.ipynb`, Section 11 (GATNet Model Definition)

```python
class GATNet(nn.Module):
    def __init__(self, num_node_features, num_classes,
                 hidden_channels=64, heads=4, dropout_rate=0.4):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_channels=num_node_features,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout_rate,
            edge_dim=1  # ⚠️ Edge attributes provided
        )
        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=num_classes,
            heads=1,
            concat=False,
            dropout=dropout_rate,
            edge_dim=1  # ⚠️ Edge attributes provided
        )
```

**Problem:** Edge attributes (edge_dim=1) may not be properly utilized by GATv2Conv
- Attention mechanism computes edge weights independently
- Edge attributes treated as additional features, not attention weights
- Graph structure should ideally USE attention for edge weighting, not vice versa
- Results in **double-counting of edge importance**: once from similarity, once from attention

---

#### 4. **Data Imbalance Not Addressed** 🔴
**Location:** `test_3.ipynb`, Section 7-8

```python
# ✅ Dataset is balanced (based on notebook variables)
N_balanced = len(df_balanced)
y_balanced contains roughly equal True/Fake samples
```

**Issue:** While balanced during training, no information about:
- Class weights in loss function
- Stratified splits (appears to use stratify, which is good)
- Handling of True Negative vs True Positive rates
- No sensitivity analysis for class imbalance scenarios

---

#### 5. **Missing Temporal Data** 🔴
**Location:** Entire project

```python
# ❌ No timestamp extraction in pipeline
# In new_main.ipynb and test_3.ipynb:
# - No 'pub_date', 'timestamp', or time column used
# - time_values appears to be undefined in Section 9
# - Project cannot access publish_date from CSV
```

The dataset likely contains temporal information but it's never extracted or verified.

---

#### 6. **Edge Index Construction Bug - Star Graph** 🟡
**Location:** `test_3.ipynb`, Section 13 (predict_news function)

```python
# Graph construction for k=10 neighbors
edge_index_new = np.concatenate([
    np.stack([np.full(topn, center), neighbors]),     # center -> neighbors
    np.stack([neighbors, np.full(topn, center)])      # neighbors -> center
], axis=1)  # Shape: (2, 2*topn)
```

**Issue:** Undirected star topology may not be optimal
- All neighbors equally connected to center
- No neighbor-to-neighbor interactions
- Graph remains sparse and disconnected
- Limits information flow in GNN layers

---

#### 7. **Device Type Casting Error** 🟡
**Location:** `ai_engine.py`
```python
# ⚠️ Questionable type conversion
data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr).to(str(device))
```

**Issue:** `.to(str(device))` converts device to string instead of proper device object
- Works by accident (PyTorch interprets string "cuda"/"cpu")
- Should use: `.to(device)` directly

---

### **CODE QUALITY ISSUES:**

#### 8. **Inconsistent Embedding Dimensions**
- WangchanBERTa: 768-dim
- XLM-RoBERTa: 768-dim (assumed)
- Different batch sizes used (32 for API, unspecified in notebooks)

#### 9. **Magic Numbers Throughout**
```python
k = 10                      # Why 10? No justification
hidden_channels = 256       # Why 256?
heads = 4                   # Why 4 heads?
alpha = 0.00135             # Why this decay constant?
max_length = 256            # Why 256 tokens?
```

#### 10. **No Hyperparameter Tuning**
- No grid search or random search
- No validation curves for hyperparameter selection
- Learning rates, dropout rates hardcoded

#### 11. **Missing Ablation Studies**
- No comparison: GCN vs GCN+Temporal
- No comparison: GCN vs GAT (GAT should outperform if attention helps)
- No comparison: with/without edge attributes
- No analysis of k neighbors impact

---

## ⚠️ WHY GAT IS NOT VERY HELPFUL

### **Fundamental Issues:**

### 1. **Star Graph Topology Limitation**
```
Problem: Star graphs have limited expressiveness
  O (center/query)
 /|\
I-O-I (neighbors only connect to center, not each other)
 \|/
  I

Impact:
- Information cannot flow between neighbors
- No mutual neighbor reinforcement
- 2-layer GAT = neighbors can only reach each other through center
- Limits pattern detection to "center node context from neighbors"
```

### 2. **Attention Redundancy with Edge Weights**
```python
# Current approach: Double-weighting
similarity_weight = 1 - cosine_distance     # Weight edges at construction
edge_attr = similarity_weight               # Pass to GNN
attention_weights = GAT(...)               # GNN learns attention AGAIN

Problem:
- If similarity already captures edge importance, GAT attention is redundant
- GAT cannot learn fundamentally different importance measure
- Attention forced to reproduce already-encoded similarity
- No room for learning non-obvious neighbor importance
```

### 3. **Limited Training Signal for Attention**
```
Binary Classification (True/Fake) Task:
- Only 2 classes → limited gradient signal
- For k=10 neighbors on 2-class problem:
  - Positive example: "all 10 neighbors predict True"
  - Negative example: "all 10 neighbors predict Fake"
- Insufficient signal to learn fine-grained attention patterns

Compare to:
- Image classification (1000 classes): Rich attention patterns needed
- NLP tasks (large vocabulary): Nuanced token importance
- Fake news detection (2 classes): Crude yes/no decision
```

### 4. **Graph Structure Doesn't Match Problem**
```
Assumption: "Similar news should have similar labels"
Reality (especially in news):
- Satirical news mimics real news format → high similarity but opposite label
- Conspiracy news shares keywords with real exposés → high similarity, different truth
- Breaking news and updates about same event → high similarity, same label

GAT learns to ignore misleading similarities, but:
- Limited examples during training (few hard cases per batch)
- Attention on 2-class problem has weak gradient signal
- Star graph prevents learning "how neighbors relate to each other"
```

### 5. **Empirical Validation Missing**
```
From test_3.ipynb analysis:
✓ The notebook has section for model comparison
✗ But actual accuracy numbers not clearly extracted
✗ No ablation: GCN vs GAT performance shown
✗ No attention weight analysis showing GAT learns meaningful patterns

If results show:
- GCN accuracy ≈ GAT accuracy → attention not helping
- GAT worse than GCN → attention hurting (worse generalization)
- GAT = GCN but slower → attention adds complexity without benefit
```

---

## ⚠️ WHY TEMPORAL DATA IS NOT VERY HELPFUL

### **Core Problems:**

### 1. **Temporal Information Not Actually Used**
```python
# ❌ time_values NEVER DEFINED or EXTRACTED
# In test_3.ipynb Section 9:
alpha = 0.00135
dt_days = abs(int(time_values[i]) - int(time_values[j])) / 86400  # time_values = ???

# Likely fallback: Uses array indices as timestamps
# This creates FAKE temporal signal:
# - News #1 and #2 are "similar in time" only because sequential in dataset
# - Not related to actual publication dates
```

### 2. **Temporal Decay Doesn't Match Domain**
```
Hypothesis: "Recent news is more similar than old news"

Reality for fake news detection:
- Conspiracy repeats across months/years (old news relevance)
- Breaking news creates sudden similar reports (burst pattern)
- Seasonal fake news (election, holidays) recurs yearly
- Exponential decay assumes linear aging of information

Actual news characteristics:
- Fake news patterns are recurring, not decaying
- Real-time fact-checking more important than temporal proximity
- Clusters of fake news around events, not continuous gradual spread
```

### 3. **Temporal Features Conflict with Similarity**
```python
# Current combination:
final_weight = similarity × temporal_decay

Problem: Conflicts
---
Scenario 1: Old identical fake news
- Similarity HIGH (same content)
- Temporal decay LOW (old)
- Confused signal: predict both high and low importance

Scenario 2: Recent false claim
- Similarity HIGH (new variant of same claim)
- Temporal decay HIGH (recent)
- Both agree, but if temporal decay too aggressive, dampens similar news

Optimal: Multi-headed approach
- Separate branches for semantic vs temporal
- Learn to weight them differently per example
- Not just multiplicative combination
```

### 4. **Dataset Temporal Distribution Unknown**
```python
# No analysis shown:
- Are all news from same time period? (no temporal variation)
- Are fake/real news temporally clustered? (temporal leakage)
- What's the time span of the dataset? (hours? months? years?)
- Is there temporal distribution mismatch between train/test?

Impact:
- If all news from same month → temporal decay ≈ all neighbors ≈ 1.0
- No real temporal signal to learn from
- Hyperparameter α becomes meaningless
```

### 5. **No Validation of Temporal Assumption**
```python
# Missing experiments:
✗ Accuracy with temporal weighting ON vs OFF
✗ Different temporal decay constants (α)
✗ Different neighborhood sizes (k)
✗ Analysis: Do temporally close news actually have same label?

# If missing, claims about temporal importance are unvalidated
```

---

## 📉 WHY OVERALL ACCURACY IS REDUCED

### **Chain of Contributing Factors:**

### 1. **Overcomplicated Architecture for 2-Class Problem**
```
Complexity Hierarchy:
- TF-IDF + LR: Simple, stable, O(n)
- Embeddings + RF: Medium, robust, O(n log n) tree operations
- BERT + LR: Complex, high-dim, O(n²) similarity matrix
- BERT + kNN + GAT: Very complex, multiple stages, cumulative error

Error Propagation:
- BERT embedding errors → kNN retrieves wrong neighbors
  → Graph construction uses wrong seeds
  → GAT trained on biased neighborhood
  → Final prediction compound error from all 3 stages
```

### 2. **kNN Star Graph Curse**
```python
# Problem: Assuming k-nearest neighbors are good evidence
# In fake news: neighbors may be:
- Unrelated content (high embedding similarity ≠ same truthfulness)
- Different language varieties (Thai regional dialects mislead similarity)
- Coordinated disinformation (intentionally mimicking real news)

Example:
- Query: "Government announces economic policy"
- Top 5 neighbors: 3 real, 2 fake (by chance)
- Star graph includes both
- GAT cannot distinguish which set is relevant
- Predicted label inherits neighbor noise
```

### 3. **BERT Embeddings May Overfit to Training Distribution**
```
WangchanBERTa trained on:
- General Thai text (not fake news specific)
- May not distinguish conspiracy language patterns
- May miss subtle lexical cues of disinformation

Example:
- BERT treats "miracle cure" embeddings -> similar to "scientific breakthrough"
- Both semantically about positive health claims
- But context (fake vs real) invisible to BERT
```

### 4. **Graph Size × Sparsity Trade-off**
```
k=10 neighbors per node:
- If dataset has 1000 news items
- Each node has 10 edges
- Graph density: 10 / 999 ≈ 0.01% (very sparse)

Impact:
- GNN convolution on sparse graph: weak signal propagation
- Information barely flows beyond immediate 1-hop neighbors
- Deeper networks (2+ layers) see limited new information
- Underfitting in graph convolutions
```

### 5. **No Class-Aware Sampling**
```python
# Current kNN: Retrieves k neighbors by similarity
# Better approach: Stratified sampling
# - Sample k neighbors from neighbors with same label (if known)
# - Sample k neighbors from neighbors with different label (contrastive)

Current problem:
- If query = Fake, but 8/10 neighbors = Real
- Graph represents mixed signal
- Binary classifier forced to average contradictory signals
```

### 6. **Training-Testing Distribution Mismatch**
```
Training procedure (from notebook):
- Build full graph on all data
- Train GNN with masks
- kNN trees built on full data

Testing procedure:
- New query → embed → kNN find neighbors → build small graph
- ⚠️ Different setting from training!
  - Training: large graph, indirect neighborhoods
  - Testing: small star graph, direct neighborhoods
  - Distribution shift → performance drop
```

### 7. **Temporal Decay with Undefined Data**
```python
# If time_values undefined or dummy:
# - Temporal decay all ≈ 1.0 (no effect)
# - Extra noise without information
# - Model focuses on fitting noise
# - Reduced generalization → lower test accuracy
```

### 8. **Gradient Issues and Training Instability**
```python
# Multi-stage pipeline:
# BERT embeddings → normalized → kNN → graph → GAT
# Gradients must flow through all stages:
# - BERT fixed (no training)
# - kNN non-differentiable (discrete neighbors)
# - Graph fixed (no meta-learning)
# - Only GAT learns

Problem:
- Information bottleneck: only final layer learns
- Fine-tuning not possible for embeddings
- No end-to-end optimization
```

### 9. **Attention Overhead with Limited Benefit**
```
GCN (2 layer):
- Conv1: (768 → 256) + ReLU
- Conv2: (256 → 2)
- Simple aggregation → stable gradients

GAT (2 layer with heads=4):
- Conv1: (768 → 64×4=256) + ReLU
- Conv2: (256 → 2)
- Multi-head attention → more parameters
- More parameters + limited data → overfitting Risk

Result:
- GAT trained with higher dropout (0.4)
- Aggressive regularization → underfitting on small graphs
- Reduced capacity
```

### 10. **Lack of Hard Negative Mining**
```
Current approach:
- Random train/val/test split
- All samples treated equally

Better approach:
- Identify hard examples (misclassified by baselines)
- Focus training on hard examples
- Use importance weighting

Without this:
- Model learns average case well
- Fails on corner cases
- Overall accuracy lower than potential
```

---

## 🔍 SPECIFIC CODE ISSUES

### **Issue 1: Undefined time_values**
**File:** `test_3.ipynb` - Section 9, line in temporal graph building
```python
dt_days = abs(int(time_values[i]) - int(time_values[j])) / 86400
# ❌ time_values never assigned or extracted
```

**Fix:**
```python
# Extract publish dates from dataframe
if 'publish_date' in df_balanced.columns:
    time_values = pd.to_datetime(df_balanced['publish_date']).astype(int) // 10**9  # Unix timestamp
else:
    print("⚠️ WARNING: No temporal data found, using dummy timestamps")
    time_values = np.arange(len(df_balanced))
```

---

### **Issue 2: Mean Pooling Including Padding**
**File:** `ai_engine.py` - Original code
```python
# ❌ Bug: Includes [PAD] tokens in mean
emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
```

**Fixed in api.py** ✅
```python
# Correct: Masks padding tokens before averaging
last_hidden = outputs.last_hidden_state
attn = inputs['attention_mask'].unsqueeze(-1)
summed = (last_hidden * attn).sum(dim=1)
denom = attn.sum(dim=1).clamp(min=1)
content_emb = (summed / denom).cpu().numpy()[0]
```

---

### **Issue 3: Device Type Casting**
**File:** `ai_engine.py`
```python
data = Data(...).to(str(device))  # ⚠️ Wrong but works
```

**Should be:**
```python
data = Data(...).to(device)  # Correct
```

---

### **Issue 4: No Validation of Graph Structure**
**File:** `test_3.ipynb`
```python
# Missing verification
# Should check:
edge_index_st.shape  # Expected (2, 2*k*N)
edge_weight_st.min(), edge_weight_st.max()  # Should be ~ [0, 1]
data_st.num_edges, data_st.num_nodes  # Sanity check
```

---

## 🛠️ Recommendations for Improvement

### **Short-term (Bug Fixes):**
1. ✅ **Fix mean pooling** - Use attention-masked averaging (already in api.py!)
2. 🔴 **Extract temporal data** - Parse publish dates from CSV
3. 🔴 **Verify time_values** - Add fallback and validation
4. 🔴 **Document magic numbers** - Add justification for k, alpha, etc.

### **Medium-term (Architecture Improvements):**
1. 🔴 **Add ablation studies:**
   - Compare GCN vs GAT accuracy
   - Temporal weighting ON vs OFF
   - Different k values (5, 7, 10, 15, 20)
   - Different temporalα decay constants

2. 🔴 **Improve graph structure:**
   - Add neighbor-to-neighbor edges (not just star)
   - Use contrastive learning (different label neighbors)
   - Implement heterogeneous graphs (real vs fake node types)

3. 🔴 **Better temporal modeling:**
   - Extract actual publish dates
   - Analyze temporal distribution
   - Use separate temporal embedding branch
   - Compare with dedicated temporal encoders

4. 🔴 **Hybrid loss function:**
   - Class-weighted BCE
   - Contrastive loss for similar vs dissimilar pairs
   - Focal loss for hard negatives

### **Long-term (Research Directions):**
1. 🟡 **Attention visualization** - Show what GAT attends to
2. 🟡 **Interpretability** - Explain which neighbors influenced decision
3. 🟡 **Domain adaptation** - How well generalizes to new sources?
4. 🟡 **Adversarial robustness** - Test against adversarial inputs
5. 🟡 **Multimodal fusion** - Combine text + images + engagement metrics

---

## 📈 Expected Performance Baseline

Based on the architecture, reasonable expectations:
- **TF-IDF + LR:** 75-80% (simple but effective)
- **BERT + RF:** 80-85% (better embeddings)
- **XLM-RoBERTa + LR:** 78-82% (multilingual but less Thai-specific)
- **GCN + BERT + Temporal:** 82-87% (should beat BERT+RF if properly tuned)
- **GAT + BERT + Temporal:** 83-88% (marginal improvement over GCN if at all)

**If observed results are lower, likely causes:**
1. ❌ Temporal data not actually used (reduces signal)
2. ❌ GAT overfitting due to limited examples
3. ❌ kNN retrieving wrong neighbors (BERT embedding mismatch)
4. ❌ Inconsistent preprocessing between train and test

---

## 📝 Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| **Data Processing** | ✅ Good | Thai normalization solid |
| **Embedding Quality** | ✅ Good | WangchanBERTa appropriate choice |
| **Graph Construction** | 🟡 Fair | Star topology limiting |
| **GAT Implementation** | 🟡 Fair | By itself, edge attributes redundant with attention |
| **Temporal Modeling** | ❌ Non-functional | time_values undefined, not actually using real timestamps |
| **Training Stability** | 🟡 Fair | Multiple stages, hard to debug |
| **Evaluation** | ✅ Good | Comprehensive metrics and error analysis |
| **Code Quality** | 🟡 Fair | Magic numbers, undocumented decisions |
| **Documentation** | ✅ Good | Well-commented Thai language code |

---

## 🎓 Key Takeaway

**Why is accuracy reduced?** The project suffers from:
1. **Over-engineering** for a 2-class problem (GAT unnecessary complexity)
2. **Missing temporal signal** (time_values undefined)
3. **Redundant edge weighting** (similarity + attention = double-counted)
4. **Limited training signal** for attention learning (only 2 classes)
5. **Sparse graph structure** limiting information flow
6. **Distribution mismatch** between training graph and test graph

**Why GAT not helpful?**
- Star graph topology prevents neighbor-to-neighbor learning
- Attention redundant with pre-computed similarity weights
- Limited gradient signal for 2-class problem
- Adds complexity without benefit validation

**Why temporal data not helpful?**
- Not actually extracted from dataset (time_values undefined)  
- Temporal decay constants arbitrary without ablation
- Domain assumptions (linear aging) don't match fake news patterns
- Conflicts with semantic similarity signals

**Recommendation:** Validate assumptions with ablations before adding complexity.

