# SLO Modeling and Cost Formulation

This section describes the latency SLO model, the separate cost functions for CARROT vs. our hardware-aware router, and the final routing score formulation used in the λ-sweep evaluation.

---

## 1. Prefill-Driven TTFT SLO

The Time-To-First-Token (TTFT) is dominated by the **prefill** phase of transformer inference, which is approximately linear in the input token length \(p\).  
We fit a regression on measured TTFT values:

$$
\text{TTFT}(p) \approx a + b \cdot p.
$$

Because real serving systems experience queueing effects, GPU interference, and scheduling jitter, we apply a **20% safety margin**:

$$
\text{TTFT}_{\text{SLO}}(p)
  = 1.20 \cdot (a + b \cdot p).
$$

This gives each prompt a personalized TTFT budget that grows proportionally with the input length.

---

## 2. Decode-Speed TPOT SLO

Token Processing Time per output Token (TPOT) varies due to GPU contention and concurrent model scheduling.  
To avoid overfitting to outliers, we use the **70th percentile** as the decode-speed SLO:

$$
\text{TPOT}_{\text{SLO}}
  = \operatorname{Percentile}_{70}(\text{TPOT}).
$$

This threshold penalizes consistently slow decoding while tolerating occasional transient spikes.

---

## 3. End-to-End (E2E) Latency SLO

E2E latency must satisfy both prefill and decode requirements.  
For an output of \(d\) tokens, the E2E latency SLO is:

$$
\text{Latency}_{\text{SLO}}
  = \text{TTFT}_{\text{SLO}}(p)
  + \text{TPOT}_{\text{SLO}} \cdot d.
$$

This decomposes the total latency budget into predictable prefill and decode components.

---

# Cost Models

CARROT and our router intentionally use **completely separate cost definitions**.

- **CARROT cost** = expected *dollar* cost of generation  
- **Our cost**     = predicted *latency* under real hardware load  
- The two cost definitions are **not mixed**, ensuring a clean and fair comparison.

---

## 4. CARROT Static Cost (Dollar-Based)

CARROT predicts the expected number of output tokens  $\hat{d}$ .  
Using each model’s fixed price-per-token, CARROT’s cost is:

$$
C_{\text{CARROT}}
  = \text{PricePerToken}(m) \cdot \hat{d}.
$$

To ensure scale stability across models and λ:

$$
C_{\text{CARROT}}^{\text{norm}}
  = \frac{C_{\text{CARROT}}}{\max(C_{\text{CARROT}})}.
$$

This preserves CARROT’s **economic interpretation** while making cost comparable to latency-based cost.

---

## 5. Our Latency-Based Cost (Hardware-Aware)

Our hardware-aware predictor estimates:

- $\hat{t}_{\text{TTFT}}$ — prefill latency  
- $ \hat{t}_{\text{TPOT}} $ — decode time per token  
- $ \hat{d} $ — predicted output length  

We form the total predicted latency:

$$
\hat{t}_{\text{total}}
  = \hat{t}_{\text{TTFT}}
  + \hat{d} \cdot \hat{t}_{\text{TPOT}}.
$$

Latency is heavy-tailed, so we use **log-scaled normalization**:

$$
C_{\text{lat}}^{\text{norm}}
  = \frac{\log\!\left(1 + \hat{t}_{\text{total}}\right)}
         {\max \log\!\left(1 + \hat{t}_{\text{total}}\right)}.
$$

This yields a stable cost signal even when latencies differ by several orders of magnitude.

---

# Routing Score Functions

For a trade-off parameter  $\lambda \in [0, 1]$ :

### CARROT routing score (quality vs. dollar cost)

$$
S_{\text{CARROT}}
  = \lambda \cdot Q
  - (1 - \lambda) \cdot C_{\text{CARROT}}^{\text{norm}}.
$$

### Our hardware-aware routing score (quality vs. latency)

$$
S_{\text{Ours}}
  = \lambda \cdot Q
  - (1 - \lambda) \cdot C_{\text{lat}}^{\text{norm}}.
$$

Where:

- \(Q\) = CARROT-predicted quality score  
- CARROT uses only predicted **dollar cost**  
- Our router uses only predicted **latency cost**  
- No cross-contamination between the two

### Special cases

-  $\lambda = 1.0$  → pure quality routing  
-  $\lambda = 0.0$  → CARROT picks cheapest model, we pick fastest model  
- Intermediate λ trace the full quality-vs-latency Pareto frontier  

---

# Why This Separation Matters

CARROT optimizes **economic efficiency** (quality per dollar).  
But in real multi-LLM, multi-GPU serving systems, latency is governed by:

- GPU queueing  
- concurrent model scheduling  
- KV-cache contention  
- TTFT spikes under load  
- variable decode throughput  

Our router directly models these hardware-level signals, making it capable of optimizing **SLO attainment**, not just cost.

This separation enables a fair head-to-head comparison:

- **CARROT** → “Which model gives best quality per dollar?”  
- **Ours**   → “Which model gives best quality per second under real hardware load?”  

This distinction reveals the advantage of hardware-aware routing in multi-tenant LLM serving.

