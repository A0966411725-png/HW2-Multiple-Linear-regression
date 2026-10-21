"""
Generic multiple-linear-regression pipeline for ANY tabular dataset (used by app.py).
CRISP-DM steps: clean -> encode -> split -> feature selection (training data only)
-> OLS -> MAE / MSE / RMSE / R² / Adj R² -> 95% prediction intervals.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import RFE, SelectKBest, SequentialFeatureSelector, f_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
METHODS = ["All-in（全部特徵）", "SelectKBest", "RFE", "Sequential FS", "Lasso", "Permutation importance"]


def guess_feature_columns(df, target):
    """Default feature list: every column except the target and obvious ID / free-text columns."""
    cols = []
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        name = str(c).strip().lower()
        if name == "id" or name.endswith("_id") or name.endswith(" id"):
            continue  # identifier column
        if not pd.api.types.is_numeric_dtype(s) and s.nunique() > 30:
            continue  # free text, dates or other high-cardinality text: one-hot would explode
        cols.append(c)
    return cols


def prepare(df, target, features):
    """Clean + encode. Numeric NA -> train median; categorical -> one-hot with drop_first (avoids the dummy trap)."""
    d = df[features + [target]].copy()
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d.dropna(subset=[target])
    cat_cols = [c for c in features if d[c].dtype == object or str(d[c].dtype) == "category" or d[c].dtype == bool]
    num_cols = [c for c in features if c not in cat_cols]
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d, num_cols, cat_cols


class Pipeline:
    """Holds everything needed to transform raw rows and predict with intervals."""

    def __init__(self, df, target, features, test_size=0.2):
        d, self.num_cols, self.cat_cols = prepare(df, target, features)
        self.target, self.features = target, features
        train, test = train_test_split(d, test_size=test_size, random_state=SEED)
        self.medians = train[self.num_cols].median()
        self.levels = {c: sorted(train[c].astype(str).fillna("missing").unique()) for c in self.cat_cols}
        self.Xtr, self.Xte = self.design(train), self.design(test)
        self.ytr, self.yte = train[target].astype(float), test[target].astype(float)
        keep = self.Xtr.columns[self.Xtr.std() > 0]  # drop constant columns
        self.Xtr, self.Xte = self.Xtr[keep], self.Xte[keep]
        self.raw_train = train
        self.n_rows_dropped = len(df) - len(d)

    def design(self, frame):
        X = frame[self.num_cols].apply(pd.to_numeric, errors="coerce").fillna(self.medians)
        parts = [X]
        for c in self.cat_cols:
            v = pd.Categorical(frame[c].astype(str).fillna("missing"), categories=self.levels[c])
            parts.append(pd.get_dummies(v, prefix=c, drop_first=True).astype(float).set_index(frame.index))
        return pd.concat(parts, axis=1)

    # ---------------- feature selection (training data only) ----------------
    def select(self, method, k):
        cols = np.array(self.Xtr.columns)
        k = int(min(k, len(cols)))
        if method.startswith("All-in") or k >= len(cols):
            return list(cols)
        Z = StandardScaler().fit_transform(self.Xtr)
        y = self.ytr.values
        if method == "SelectKBest":
            return list(cols[SelectKBest(f_regression, k=k).fit(Z, y).get_support()])
        if method == "RFE":
            return list(cols[RFE(LinearRegression(), n_features_to_select=k).fit(Z, y).support_])
        cv = KFold(min(5, max(2, len(y) // 10)), shuffle=True, random_state=SEED)
        if method == "Sequential FS":
            sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction="forward",
                                            cv=cv, scoring="r2").fit(Z, y)
            return list(cols[sfs.get_support()])
        if method == "Lasso":
            coef = pd.Series(np.abs(LassoCV(cv=cv, random_state=SEED).fit(Z, y).coef_), index=cols)
            return list(coef.sort_values(ascending=False).index[:k])
        # permutation importance on a hold-out part of the training set
        Za, Zb, ya, yb = train_test_split(Z, y, test_size=0.25, random_state=SEED)
        imp = permutation_importance(LinearRegression().fit(Za, ya), Zb, yb, n_repeats=10, random_state=SEED)
        return list(pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False).index[:k])

    # ---------------- model + evaluation ----------------
    def fit(self, cols):
        self.cols = cols
        self.model = sm.OLS(self.ytr, sm.add_constant(self.Xtr[cols], has_constant="add")).fit()
        return self.model

    def evaluate(self):
        sf = self.predict_frame(self.Xte)
        yp, yt = sf["mean"].values, self.yte.values
        n, k = len(yt), len(self.cols)
        r2 = r2_score(yt, yp)
        inside = (yt >= sf["obs_ci_lower"].values) & (yt <= sf["obs_ci_upper"].values)
        return {"MAE": mean_absolute_error(yt, yp), "MSE": mean_squared_error(yt, yp),
                "RMSE": float(np.sqrt(mean_squared_error(yt, yp))), "R2": r2,
                "AdjR2": 1 - (1 - r2) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else float("nan"),
                "train_R2": float(self.model.rsquared), "PI95_coverage": float(inside.mean() * 100),
                "n_train": len(self.ytr), "n_test": n}, sf

    def predict_frame(self, X, alpha=0.05):
        Xd = sm.add_constant(X[self.cols], has_constant="add")
        sf = self.model.get_prediction(Xd).summary_frame(alpha=alpha)
        sf.index = X.index
        return sf

    def predict_row(self, raw):
        return self.predict_frame(self.design(pd.DataFrame([raw]))).iloc[0]

    def compare_methods(self, k):
        """Test-set R² / RMSE of every selection method at the same k (training data only for selection)."""
        rows = []
        for m in METHODS:
            try:
                cols = self.select(m, k)
                lr = LinearRegression().fit(self.Xtr[cols], self.ytr)
                p = lr.predict(self.Xte[cols])
                rows.append({"方法": m, "特徵數": len(cols), "Test R²": r2_score(self.yte, p),
                             "Test RMSE": float(np.sqrt(mean_squared_error(self.yte, p))),
                             "選出的特徵": ", ".join(cols)})
            except Exception as e:  # e.g. too few rows for CV
                rows.append({"方法": m, "特徵數": None, "Test R²": None, "Test RMSE": None, "選出的特徵": f"無法執行：{e}"})
        return pd.DataFrame(rows)
