# 🗺️ Quick Navigation Guide to Improvements

## Notebook Structure: test_4.ipynb

### 📍 Finding Key Sections

```
SECTION 6: TEMPORAL PREPROCESSING (ENHANCED)
├─ 🔴 IMPROVED DATE PARSING
├─ Robust Thai date parsing with validation
├─ Verify time_values accuracy
└─ Location: Search for "Parse Thai Datetime Report"
   Output: Date validation dashboard + time_values verification

SECTION 9: BUILD TEMPORAL GRAPH (DOCUMENTED)
├─ 🔴 MAGIC NUMBERS EXPLAINED
│
├─ k = 10 neighbors
│  └─ Half-life: 512 days
│     Lower k (5): Sparser graph, less noise
│     Higher k (20): More connections, risk of over-smoothing
│
├─ α = 0.00135 decay rate
│  └─ Formula: half_life = ln(2) / α ≈ 512 days
│     Justify: Most news relevance expires in 1-2 years
│
├─ Edge Weight = cos_sim × exp(-α·Δt)
│  └─ Semantic × Temporal weighting
│
└─ Ablation parameters READY TO TEST: k∈{5,10,15,20}, α∈{0.0001,0.00135,0.005}

SECTION 9.5: RICHER GRAPH STRUCTURES (OPTIONAL)
├─ 🔴 NON-STAR TOPOLOGY IMPROVEMENTS
├─ Temporal bin connections (same-week grouping)
├─ Category-aware connections (within-category kNN)
├─ Combined graph merging utilities
└─ Location: Uncomment to activate

SECTION 11A: ADVANCED LOSS FUNCTIONS (NEW)
├─ 🔴 IMPROVED LOSS FUNCTIONS
├─ FocalLoss: Focus on hard examples (γ=2.0)
├─ WeightedCELoss: Balance class frequencies
└─ Usage examples + auto weight computation

SECTION 11B: IMPROVED GNN MODELS (NEW)
├─ 🔴 TIME EMBEDDING BRANCH
│
├─ GATNetWithTimeEmbedding
│  ├─ Semantic branch: BERT → GAT
│  ├─ Temporal branch: Δt → MLP → embeddings
│  └─ Fusion: Both guide attention
│
└─ GCNNetWithTemporalAblation: For baseline comparisons

SECTION 12A-C: ABLATION STUDY FRAMEWORK (NEW)
├─ 🔴 COMPREHENSIVE ABLATION INFRASTRUCTURE
│
├─ Test 5 configurations:
│  1. GAT + CE + Temporal [BASELINE]
│  2. GAT + CE (NO Temporal) ← Temporal value?
│  3. GAT + Focal + Temporal ← Loss function?
│  4. GAT + TimeEmbed + CE ← Learned time?
│  5. GCN + CE + Temporal ← Architecture?
│
├─ Automatic result compilation
├─ Summary table showing all metrics
├─ Training curves for each config
├─ Comparison bar charts
│
└─ Output: "ablation_study_full_comparison.png"

SECTION 14: ENHANCED ATTENTION VISUALIZATION (NEW)
├─ 🟡 EXPLAINABILITY IMPROVEMENTS
│
├─ plot_attention_network_graph()
│  ├─ Center: Query news (gold node)
│  ├─ Circle: Top-K similar items (colored by label)
│  ├─ Edge width: Attention strength
│  └─ Shows model reasoning
│
├─ plot_attention_heatmap_layer_wise()
│  ├─ Layer 1 attention patterns
│  ├─ Layer 2 attention patterns
│  └─ Information flow visualization
│
└─ plot_temporal_vs_attention_correlation()
   ├─ Temporal dist vs attention (correlation?)
   ├─ Semantic sim vs attention (correlation?)
   └─ Validate graph weight importance

SECTION 9 (PART 9): DOMAIN ADAPTATION & ROBUSTNESS (NEW)
├─ 🟡 ADVERSARIAL & DOMAIN TESTS
│
├─ Adversarial robustness:
│  ├─ Word swap attacks
│  ├─ Word removal attacks
│  ├─ Character swaps
│  └─ Tests: Clean acc vs perturbed acc
│
├─ Temporal distribution shift:
│  └─ Train on old, test on recent (generalization?)
│
├─ Confidence calibration:
│  ├─ Is model confidence = actual accuracy?
│  ├─ Expected Calibration Error (ECE)
│  └─ Reliability diagram
│
└─ Outputs: Robustness reports + calibration plot
```

---

## 🎯 Quick Lookup by Question

### "I want to understand magic numbers k and α"
→ **SECTION 9**, search for "**HYPERPARAMETER SELECTION**"
- k explanation: "`WHY 10?`"
- α explanation: "`WHY 0.00135?`"
- Formula box showing half-life calculation
- Ablation test ranges provided

### "How is the date parsing validated?"
→ **SECTION 6**, search for "**IMPROVED DATE PARSING**"
- Parsing quality report output
- NaT handling with fallback strategies
- Temporal range verification
- Check: Any dates outside [2010-2026]?

### "Which model performs best? Should I use GAT or GCN?"
→ **SECTION 12B-C**, search for "**ABLATION STUDY**"
- Summary table with all 5 models
- Test accuracy comparison
- F1-score, Precision, Recall per model
- Training curves for each
- Auto-selection of best performer

### "Is temporal decay really important or just hype?"
→ **SECTION 12** (Config 1 vs Config 2)
- Experiment 1: WITH temporal decay
- Experiment 2: WITHOUT temporal decay
- Compare test accuracy drop
- If drop < 2%: temporal not important
- If drop > 10%: temporal crucial

### "How do I know which neighbors influenced the prediction?"
→ **SECTION 14**, function: `plot_attention_network_graph()`
- Shows: Center node (query) + 10 neighbors in circle
- Neighbor color: Blue=real, Red=fake
- Edge width & node size: Attention weight
- Top-K ranked list of influential samples

### "Is the model robustly fooling-resistant?"
→ **SECTION 9 (PART 9)**, section: "**ADVERSARIAL ROBUSTNESS TEST**"
- Test with word swaps, removals, character modifications
- Report: Accuracy drop % for each perturbation type
- If drop < 5%: Very robust ✅
- If drop > 15%: Fragile ❌

### "If I train on old news, can I predict recent news?"
→ **SECTION 9 (PART 9)**, section: "**TEMPORAL DISTRIBUTION SHIFT TEST**"
- Split: 80% old news vs 20% recent news
- Tests if model generalizes to new temporal distribution
- Identifies temporal bias in model

### "Is the model overconfident? Can I trust its probabilities?"
→ **SECTION 9 (PART 9)**, section: "**CONFIDENCE CALIBRATION**"
- Expected Calibration Error (ECE) score
- Reliability diagram (diagonal = well-calibrated)
- ECE < 0.1 = Good ✅
- ECE > 0.2 = Overconfident ❌

---

## 🔧 How to Run Specific Experiments

### Experiment 1: Run minimal ablation (just 2 configs)
```python
# Section 12B: Comment out other experiments, keep only:
# - Config 1: GAT + CE + Temporal
# - Config 2: No Temporal

# Then just run that subsection
```

### Experiment 2: Test different k and α values
```python
# In Section 9:
k = 5  # Change here
alpha = 0.0005  # Change here

# Then rerun Section 9 → Section 10 → Section 12
```

### Experiment 3: Use learned time embeddings instead of exponential decay
```python
# In Section 12B:
# Replace this line:
model_gat_ce = GATNet(...)

# With:
model_gat_time_emb = GATNetWithTimeEmbedding(...)

# This uses learned time encoding instead of fixed exponential
```

### Experiment 4: Use focal loss for harder learning
```python
# In Section 12B, Config 1:
# Replace:
criterion_ce = nn.CrossEntropyLoss()

# With:
criterion_focal = FocalLoss(alpha=class_weights, gamma=2.0)

# Keeps same model, just different loss
```

### Experiment 5: Add richer graph structure
```python
# In Section 9.5: Uncomment the example code block
edges_temp, weights_temp = build_temporal_bin_graph(...)
edges_cat, weights_cat = build_category_aware_graph(...)

# Then create combined graph:
edges_combined, weights_combined = combine_graph_structures(...)

# Use combined edges in training
```

---

## 📊 Output Files Checklist

After running full notebook, you should see:

```
✅ ablation_study_full_comparison.png
   - 5 training curve subplots
   - 1 comparison bar chart
   - Shows: 5 experiments converging + final metrics

✅ plot1_overview_dashboard.png
   - 4-panel dashboard of edge statistics
   - Temporal distribution, similarity distribution, combined weights, scatter

✅ plot2_kde_true_vs_fake.png
   - KDE comparison: true news vs fake news
   - Temporal patterns, semantic patterns

✅ plot3_hexbin_density.png
   - 2D density heatmaps (temporal × semantic)
   - Separate for true vs fake

✅ plot4_heatmap_weight.png
   - Edge weight distribution across bins
   - Shows relationship between α and temporal decay

✅ plot5_decay_curve.png
   - Different α values visualization
   - Contour plot of final edge weight formula
```

---

## 🚨 Common Issues & Fixes

### Issue: "NaT values not being parsed"
**Solution:** Check Section 6 output
- See what % of dates parsed successfully
- If <80%: May have unexpected date format
- Check first few unparsed values: `df_balanced['วันและเวลาที่เผยแพร่'].head()`

### Issue: "Training too slow / won't converge"
**Solution:** Check ablation section
- N_balanced too large? Try 50% sample
- Epochs=200 too many? Try 100
- Learning rate too small? Try 0.002

### Issue: "Ablation Table not showing all metrics"
**Solution:** Run full Section 12B-C
- Each experiment must complete to register
- May timeout on weak hardware: reduce num_epochs to 100

### Issue: "Attention graph shows only 5 neighbors instead of 10"
**Solution:** Adjust in visualization
```python
top_k=10  # Change this parameter
plot_attention_network_graph(..., top_k=10)
```

---

## 💡 Tips & Tricks

### Speed Up Ablation:
- Use `num_epochs=100` instead of 200 (still trains well)
- Set `verbose_interval=50` to see fewer logs

### Understand Ablation Results Better:
- Plot just individual config: `plt.plot(ablation.results['GAT + CE + Temporal']['history']['train_loss'])`
- Focus on: Best Val Acc (generalization) vs Test Acc (true performance)

### Diagnose Temporal Importance:
- If Config 1 Test Acc ≈ Config 2 Test Acc: temporal not helping
  - Try larger α (faster decay)
  - Try smaller k (fewer neighbors)
  - May need richer graph structure (Section 9.5)

### Improve Robustness:
- If adversarial test shows > 20% accuracy drop
  - Try Focal Loss (Section 11A)
  - Try larger model (more hidden channels)
  - Try data augmentation (generate fake adversarial examples as training data)

---

## 🎓 Learning Path

### Beginner (understand current pipeline):
1. Read SECTION 6 (date parsing)
2. Read SECTION 9 (graph building + magic numbers explained)
3. Run SECTION 12 (ablation) once
4. Look at output table: which model wins?

### Intermediate (modify & experiment):
1. Modify k, α values in Section 9
2. Re-run ablation with new hyperparameters
3. Check if performance improves
4. Look at attention visualization (Section 14)

### Advanced (research contributions):
1. Implement Section 9.5 (richer graphs)
2. Design new graph topologies
3. Run cross-experiment comparisons
4. Publish findings (improved accuracy + interpretability)

---

## 📞 Support

**Questions about specific sections?**
- Each section has inline comments explaining code
- Search (Ctrl+F) for section number to jump to it

**Questions about improvements?**
- Read: `IMPROVEMENTS_SUMMARY.md` (comprehensive docs)
- This file: `QUICK_REFERENCE.md` (navigation)

**Found a bug?**
- Check: Date formats in CSV
- Check: Any NaT/inf values in time_values
- Check: Dimensions match: x_balanced.shape[0] == N_balanced == len(y_balanced)

---

*Last Updated: March 2026*
*Quick Reference for test_4.ipynb improvements*
