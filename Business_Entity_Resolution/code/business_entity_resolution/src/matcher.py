from typing import Dict, List, Tuple
import numpy as np
from .scorer import fbeta_single


class EntityMatcher:
    def __init__(self, threshold: float = 0.5, use_lightgbm: bool = True):
        self.threshold = threshold
        self.use_lightgbm = use_lightgbm
        self.model = None
        self._has_lgbm = False
        self._weights = None
        self._bias = 0.0
        self._try_import()

    def _try_import(self):
        if not self.use_lightgbm:
            return
        try:
            import lightgbm  # noqa: F401
            self._has_lgbm = True
        except ImportError:
            self._has_lgbm = False

    def _build_model(self, n_features: int):
        if self._has_lgbm:
            import lightgbm as lgb
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "num_leaves": 63,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.85,
                "bagging_freq": 5,
                "min_child_samples": 20,
                "reg_alpha": 0.05,
                "reg_lambda": 0.1,
                "verbose": -1,
                "n_jobs": -1,
                "seed": 42,
            }
            return ("lgb", params, None)
        else:
            return ("heuristic", {}, None)

    def _train_logreg(self, X: np.ndarray, y: np.ndarray,
                      n_iter: int = 200, lr: float = 0.1) -> None:
        n, d = X.shape
        w = np.zeros(d, dtype=np.float64)
        b = 0.0
        pos = max(y.sum(), 1)
        neg = max((1 - y).sum(), 1)
        weight = np.where(y == 1, neg / (pos + neg), pos / (pos + neg))
        weight = weight * 2.0
        for it in range(n_iter):
            logits = X @ w + b
            logits = np.clip(logits, -30, 30)
            p = 1.0 / (1.0 + np.exp(-logits))
            err = (p - y) * weight
            grad_w = X.T @ err / n + 0.001 * w
            grad_b = err.mean()
            w -= lr * grad_w
            b -= lr * grad_b
        self._weights = w.astype(np.float32)
        self._bias = float(b)

    def _heuristic_predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._weights is None:
            weights = np.array([
                2.0, 1.2, 1.0, 0.6, 0.6,
                2.5, 1.5, 1.2, 1.0, 1.2,
                1.5, 0.8, 0.5, 0.3, 0.8,
                0.2, -0.5, 0.0, 1.5,
                0.0, 0.0, 0.0,
                1.5, 1.0, 0.8, 0.4, 0.4,
                2.0, 1.2, 1.0, 0.8, 1.0,
                1.0, 0.5, 0.3, 0.2, 0.5,
                0.15, -0.5, 0.0, 1.5,
                0.0, 0.0, 0.0,
                1.8, 0.8, 0.5, 3.0, 2.0, 4.0, 4.0,
            ], dtype=np.float32)
            if len(weights) < X.shape[1]:
                weights = np.pad(weights, (0, X.shape[1] - len(weights)))
            else:
                weights = weights[:X.shape[1]]
            self._weights = weights
            self._bias = -2.5
        logits = X @ self._weights + self._bias
        logits = np.clip(logits, -30, 30)
        proba = 1.0 / (1.0 + np.exp(-logits))
        return proba.astype(np.float32)

    def train(self, X: np.ndarray, y: np.ndarray, val_X: np.ndarray = None,
              val_y: np.ndarray = None):
        n_features = X.shape[1]
        self._model_info = self._build_model(n_features)
        mtype = self._model_info[0]
        if mtype == "lgb" and self._has_lgbm:
            import lightgbm as lgb
            params = self._model_info[1].copy()
            lgb_train = lgb.Dataset(X, y)
            val_sets = [lgb_train]
            if val_X is not None and val_y is not None:
                lgb_val = lgb.Dataset(val_X, val_y, reference=lgb_train)
                val_sets.append(lgb_val)
            num_boost_round = 1000
            self.model = lgb.train(
                params, lgb_train,
                num_boost_round=num_boost_round,
                valid_sets=val_sets,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            print(f"[matcher] LightGBM trained, best_iter={self.model.best_iteration}")
        else:
            self.model = "heuristic"
            if len(X) > 0 and len(np.unique(y)) > 1:
                self._train_logreg(X, y)
                print(f"[matcher] Heuristic (trained logistic regression) "
                      f"{len(X)} samples, pos={int(y.sum())}")
            else:
                self._weights = None
                print("[matcher] Heuristic (default weights)")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0, dtype=np.float32)
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        if self._has_lgbm and self.model != "heuristic":
            return self.model.predict(X, num_iteration=self.model.best_iteration)
        return self._heuristic_predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(np.int32)

    def tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray,
                       val_pairs_info: List[Tuple[str, str, Dict[str, set]]],
                       beta: float = 0.5) -> float:
        proba = self.predict_proba(X_val)
        s1_ids = []
        for s1_id, _, _ in val_pairs_info:
            if s1_id not in s1_ids:
                s1_ids.append(s1_id)
        gt_by_s1: Dict[str, set] = {}
        cand_by_s1: Dict[str, List[Tuple[float, str]]] = {}
        idx = 0
        for s1_id, cid, yt in val_pairs_info:
            if s1_id not in gt_by_s1:
                gt_by_s1[s1_id] = yt if yt is not None else set()
                cand_by_s1[s1_id] = []
            cand_by_s1[s1_id].append((proba[idx] if idx < len(proba) else 0.0, cid))
            idx += 1
        best_t = self.threshold
        best_f = -1.0
        for t in np.arange(0.05, 0.96, 0.01):
            f_vals = []
            for s1_id in s1_ids:
                gt = gt_by_s1.get(s1_id, set())
                pred = set(cid for sc, cid in cand_by_s1.get(s1_id, []) if sc >= t)
                f_vals.append(fbeta_single(gt, pred, beta))
            avg_f = float(np.mean(f_vals)) if f_vals else 0.0
            if avg_f > best_f:
                best_f = avg_f
                best_t = t
        self.threshold = float(best_t)
        print(f"[matcher] Tuned threshold={self.threshold:.3f} val_F{beta}={best_f:.4f}")
        return self.threshold
