# algoRhythmss: Dual-Objective PCAM Precision Engine
**Team:** algoRhythmss (Thejas J, Raghavendra, Abhijith, Dhul Kif)

## 1. Architecture Overview
Our agent utilizes a confidence-gated, dual-objective architecture to control the Predictive Coding Associative Memory (PCAM) dynamics without neural network retraining. We explicitly designed this engine to be **Scale-Invariant**, dynamically adapting to high-dimensional real-world datasets (like PCA-MNIST) evaluated in L3.

## 2. Retrieval: Variance-Adaptive Noise Masking
For high-noise queries, we engineered a variance-ratio controller: $\Pi_{ret} = Var(landscape) / Var(noise)$. 
Instead of hardcoding cosine thresholds or dimensions, we estimate the expected clean state using scale-invariant soft-attention. By calculating the ratio of the landscape's natural variance against the query's specific squared error, our agent dynamically crushes the precision of highly corrupted dimensions. This zero-shot approach achieved a massive **+0.115 $\Delta$ accuracy** over the baseline.

## 3. Anisotropy: True Equilibrium Preconditioning (Lemma E3 + Theorem F3)
Through spectral analysis, we identified that the synthetic $R$ matrix contains a uniform diagonal, rendering it mathematically invariant to diagonal preconditioning. However, real-world datasets exhibit massive variance skew across dimensions.

To isotropise this skewed geometry for the L3 evaluation, we bypassed heavy offline optimizers (like CMA-ES) in favor of closed-form linear algebra:
1. **Lemma E3:** We approximate the true equilibrium as $a^* \approx \eta R^{-1} x_i$ to capture the true intra-cluster covariance matrix.
2. **Theorem F3:** We evaluate the exact Hessian at $a^*$ and apply the Jacobi Preconditioner ($\Pi = 1 / diag(H)$). 

This reduces offline initialization compute to $O(KN)$ and online inference to $O(KN)$, guaranteeing microsecond latency during evaluation while maximizing spread reduction on clustered datasets.

## 4. Reproducibility
```bash
pip install -r requirements.txt
python self_check.py --adapter adapters.algorythms:Engine
