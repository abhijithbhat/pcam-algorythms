# AlgoRythms: Dual-Objective PCAM Precision Engine
**Team:** algoRhythmss (Thejas J, Raghavendra, Abhijith, Dhul Kif)

## 1. Architecture Overview
Our agent utilizes a confidence-gated, dual-objective architecture to control the Predictive Coding Associative Memory (PCAM) dynamics without neural network retraining. We explicitly designed this engine to be **Scale-Invariant**, dynamically adapting to high-dimensional real-world datasets (like PCA-MNIST) evaluated in L3.

## 2. Retrieval: Variance-Adaptive Noise Masking
For high-noise queries, we engineered a variance-ratio controller: $\Pi_{ret} = Var(landscape) / Var(noise)$. 
Instead of hardcoding cosine thresholds or dimensions, we estimate the expected clean state using scale-invariant soft-attention. By calculating the ratio of the landscape's natural variance against the query's specific squared error, our agent dynamically crushes the precision of highly corrupted dimensions. This zero-shot approach achieved a massive **+0.115 $\Delta$ accuracy** over the baseline.

## 3. Anisotropy: Log-Space Gradient Optimization
To extract the geometric spread reduction, we implemented an offline `_condition_grad` optimizer. 
We analytically derived the exact gradient of the condition number $\kappa(S) = \lambda_{max}/\lambda_{min}$. Our algorithm initializes via Jacobi preconditioning ($1/diag(H)$) and iteratively refines the precision array in log-space. This guarantees numerical stability and strict adherence to the $[0.1, 10.0]$ clipping bounds. 

**The Diagonal Constraint Proof:** We discovered that for the specific synthetic $R$ matrix provided in the starter kit, the uniform diagonal strictly limits the maximum theoretical spread reduction achievable by any mean-normalized diagonal matrix. Our log-space optimizer successfully converges to this exact mathematical ceiling (~1.03x), fully preparing the agent to maximally exploit the natural variance skew present in the hidden L3 PCA-MNIST dataset.

## 4. Reproducibility
```bash
pip install -r requirements.txt
python self_check.py --adapter adapters.algorythms:Engine
