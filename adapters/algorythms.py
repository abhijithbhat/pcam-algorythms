import numpy as np
from adapter import Adapter


_PI_LO, _PI_HI = 0.1, 10.0
_EPS = 1e-12
_SMOOTH_FLOOR = 0.05

_LOG_ITERS = 20
_LOG_ALPHAS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)
_EIGEN_MODES = (0, 1, 2, -1)


class Engine(Adapter):
    def __init__(self, stored_patterns, model_params):
        super().__init__(stored_patterns, model_params)
        self.X = np.array(stored_patterns, dtype=np.float64)
        self.K, self.D = self.X.shape
        self.R = np.array(model_params["R"], dtype=np.float64)
        self.eta = float(model_params["eta"])
        self.beta = float(model_params["beta"])
        self.pi_min = float(model_params.get("pi_min", _PI_LO))
        self.pi_max = float(model_params.get("pi_max", _PI_HI))
        self.X_norms = np.linalg.norm(self.X, axis=1) + _EPS
        self.R_diag = np.diag(self.R).copy()

        self._iso_pis = np.empty((self.K, self.D), dtype=np.float64)
        for k in range(self.K):
            self._iso_pis[k] = self._optimise_iso_pi(self.X[k], seed=k)

    def _compute_hessian(self, a):
        z = self.beta * (self.X @ a)
        z -= z.max()
        e = np.exp(z)
        s = e / (e.sum() + _EPS)
        soft_cov = np.diag(s) - np.outer(s, s)
        H = self.R - self.eta * self.beta * (self.X.T @ soft_cov @ self.X)
        return 0.5 * (H + H.T)

    def _project(self, pi):
        pi = np.asarray(pi, dtype=np.float64)
        pi = np.maximum(pi, _EPS)
        pi = np.clip(pi, self.pi_min, self.pi_max)
        pi = pi / (pi.mean() + _EPS)
        return pi

    def _spread(self, H, pi):
        pi = self._project(pi)
        ps = np.sqrt(pi)
        S = (ps[:, None] * H) * ps[None, :]
        ev = np.linalg.eigvalsh(0.5 * (S + S.T))
        ev = ev[ev > 1e-9]
        return ev[-1] / ev[0] if len(ev) >= 2 else 1e12

    def _pi_from_log(self, x):
        x = np.asarray(x, dtype=np.float64)
        x = x - x.max()
        pi = np.exp(np.clip(x, -30.0, 30.0))
        return self.D * pi / (pi.sum() + _EPS)

    def _condition_grad(self, H, x):
        pi = self._pi_from_log(x)
        ps = np.sqrt(np.maximum(pi, _EPS))
        S = (ps[:, None] * H) * ps[None, :]
        ev, vecs = np.linalg.eigh(0.5 * (S + S.T))

        lam_min = max(ev[0], _EPS)
        lam_max = ev[-1]
        v_min = vecs[:, 0]
        v_max = vecs[:, -1]
        d_max = v_max * (H @ (ps * v_max)) / (ps + _EPS)
        d_min = v_min * (H @ (ps * v_min)) / (ps + _EPS)
        grad_pi = d_max / lam_min - lam_max * d_min / (lam_min * lam_min)
        grad_x = pi * (grad_pi - (pi * grad_pi).sum() / self.D)
        return lam_max / lam_min, grad_x, pi

    def _try_candidate(self, H, best, raw):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (self.D,) or not np.all(np.isfinite(raw)):
            return best
        pi = self._project(raw)
        sp = self._spread(H, pi)
        return (sp, pi.copy()) if sp < best[0] else best

    def _optimise_iso_pi(self, pattern, seed=0):
        H = self._compute_hessian(pattern)
        vals, vecs = np.linalg.eigh(H)
        if vals[0] <= 0:
            return np.ones(self.D)

        best = (self._spread(H, np.ones(self.D)), np.ones(self.D))

        diag = np.maximum(np.diag(H), _EPS)
        row_abs = np.maximum(np.sum(np.abs(H), axis=1), _EPS)
        off_abs = row_abs - np.abs(np.diag(H))
        gersh = np.maximum(np.diag(H) - off_abs, _EPS)
        for base in (diag, row_abs, gersh):
            best = self._try_candidate(H, best, 1.0 / base)
            best = self._try_candidate(H, best, 1.0 / np.sqrt(base))
            best = self._try_candidate(H, best, base)

        logits = self.beta * (self.X @ pattern)
        logits -= logits.max()
        p = np.exp(logits)
        p = p / (p.sum() + _EPS)
        mu = p @ self.X
        local_var = p @ ((self.X - mu) ** 2)
        h_diag = np.maximum(self.R_diag - self.eta * self.beta * local_var, _EPS)
        best = self._try_candidate(H, best, 1.0 / h_diag)
        best = self._try_candidate(H, best, 1.0 / np.sqrt(h_diag))

        starts = [np.log(np.maximum(best[1], 1e-8)), np.zeros(self.D)]
        for col in _EIGEN_MODES:
            mode = vecs[:, col] ** 2
            mode = mode / (mode.mean() + _EPS)
            starts.append(np.log(np.maximum(np.clip(mode, self.pi_min, self.pi_max), 1e-8)))

        rng = np.random.default_rng(seed + 99)
        starts.append(0.05 * rng.standard_normal(self.D))

        for start in starts:
            x = start.copy()
            for _ in range(_LOG_ITERS):
                f, grad, pi = self._condition_grad(H, x)
                sp = self._spread(H, pi)
                if sp < best[0]:
                    best = (sp, pi.copy())

                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-10:
                    break

                direction = -grad / (grad_norm + _EPS)
                moved = False
                for alpha in _LOG_ALPHAS:
                    f_try, _, _ = self._condition_grad(H, x + alpha * direction)
                    if f_try < f - 1e-12:
                        x = x + alpha * direction
                        moved = True
                        break
                if not moved:
                    break

        return np.clip(best[1], self.pi_min, self.pi_max)

    def predict_precision(self, corrupted_query):
        q = np.asarray(corrupted_query, dtype=np.float64)
        q_norm = np.linalg.norm(q) + _EPS
        cos_sim = self.X @ q / (self.X_norms * q_norm)
        best_k = int(np.argmax(cos_sim))
        max_cos = cos_sim[best_k]

        # 1. Scale-Invariant Attention (adapts to any dimension)
        logits = np.sqrt(self.D) * cos_sim
        logits -= logits.max()
        w = np.exp(logits)
        w = w / (w.sum() + _EPS)
        
        mu = w @ self.X
        local_var = w @ ((self.X - mu) ** 2)
        noise_sq = (q - mu) ** 2
        pi_ret = (local_var + _SMOOTH_FLOOR) / (noise_sq + _SMOOTH_FLOOR)

        # 2. Dynamic Statistical Gating (adapts to L3 PCA-MNIST)
        mean_cos = cos_sim.mean()
        std_cos = cos_sim.std() + _EPS
        
        dynamic_centre = mean_cos + 2.5 * std_cos
        dynamic_pure = mean_cos + 3.0 * std_cos
        dynamic_steepness = 5.0 / std_cos

        pi_iso = self._iso_pis[best_k]
        
        # Pure ISO fallback
        if max_cos >= dynamic_pure:
            return pi_iso.copy()

        # 3. Geometric Blend
        t = 1.0 / (1.0 + np.exp(-dynamic_steepness * (max_cos - dynamic_centre)))
        pi = np.exp(t * np.log(pi_iso + _EPS) + (1.0 - t) * np.log(pi_ret + _EPS))
        
        # 4. Rigorous Constraint Projection
        pi = pi / (pi.mean() + _EPS)
        for _ in range(20):
            pi = np.clip(pi, self.pi_min, self.pi_max)
            mean = pi.mean()
            if abs(mean - 1.0) < 1e-10:
                break
            pi = pi / (mean + _EPS)
            
        return np.clip(pi, self.pi_min, self.pi_max)
