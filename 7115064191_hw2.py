"""
HW2 - Multiple Linear Regression on King County House Sales (CRISP-DM)
Student ID : 7115064191
Dataset    : House Sales in King County, USA (Kaggle)
             https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
Run        : python 7115064191_hw2.py   (Mac: python3; ~1 min)
Outputs    : figures/*.png, results/*.csv|json, model/house_price_mlr.pkl
"""
import json
import os
import pickle
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

warnings.filterwarnings("ignore")
SEED = 42
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "kc_house_data.csv")
FIG = os.path.join(BASE, "figures")
RES = os.path.join(BASE, "results")
MOD = os.path.join(BASE, "model")
for d in (FIG, RES, MOD):
    os.makedirs(d, exist_ok=True)


class Tee:
    """Print to the console and to results/run_log.txt at the same time (UTF-8 on Mac and Windows)."""
    def __init__(self, path):
        self.console = sys.stdout
        self.file = open(path, "w", encoding="utf-8")

    def write(self, s):
        try:
            self.console.write(s)
        except UnicodeEncodeError:  # e.g. Windows cp950 console cannot print "²"
            self.console.write(s.encode(self.console.encoding or "ascii", "replace").decode(self.console.encoding or "ascii"))
        self.file.write(s)

    def flush(self):
        self.console.flush()
        self.file.flush()


sys.stdout = Tee(os.path.join(RES, "run_log.txt"))

# ---- chart style (validated reference palette) ----
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#ecebe6", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.axisbelow": True,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print(f"  [fig] figures/{name}")


def dollar_metrics(y_true_log, y_pred_log, k=None):
    """Metrics on the original $ scale (plus R² on log scale)."""
    yt, yp = np.exp(y_true_log), np.exp(y_pred_log)
    n = len(yt)
    r2_log = r2_score(y_true_log, y_pred_log)
    out = {
        "R2_log": r2_log,
        "R2_$": r2_score(yt, yp),
        "RMSE_$": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE_$": mean_absolute_error(yt, yp),
        "MAPE_%": float(np.mean(np.abs((yt - yp) / yt)) * 100),
    }
    if k is not None:
        out["AdjR2_log"] = 1 - (1 - r2_log) * (n - 1) / (n - k - 1)
    return out


# =====================================================================
# 1. Business Understanding
# =====================================================================
print("=" * 70)
print("1. BUSINESS UNDERSTANDING")
print("=" * 70)
print("Goal : predict house sale price in King County (Seattle area) so that")
print("       buyers / sellers / agents can get a fair price estimate with an")
print("       uncertainty range (95% prediction interval).")
print("KPI  : test R², RMSE, MAE, MAPE; PI coverage close to 95%.")

# =====================================================================
# 2. Data Understanding
# =====================================================================
print("\n" + "=" * 70)
print("2. DATA UNDERSTANDING")
print("=" * 70)
df = pd.read_csv(DATA)
print(f"shape = {df.shape}")
FEATURES_RAW = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
                "waterfront", "view", "condition", "grade", "sqft_above",
                "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat",
                "long", "sqft_living15", "sqft_lot15"]
print(f"features = {len(FEATURES_RAW)} (+ id, date, target price)")
print(f"missing values = {int(df.isna().sum().sum())}, duplicated ids = {int(df['id'].duplicated().sum())}")
desc = df[["price"] + FEATURES_RAW].describe().T
desc.to_csv(os.path.join(RES, "describe.csv"))
print(desc[["mean", "std", "min", "max"]].round(2).to_string())

# price distribution: raw vs log
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].hist(df["price"] / 1e6, bins=80, color=BLUE, edgecolor="white", linewidth=0.4)
ax[0].set(title="Price (raw) - right skewed", xlabel="Price (million USD)", ylabel="Count")
ax[1].hist(np.log(df["price"]), bins=80, color=BLUE, edgecolor="white", linewidth=0.4)
ax[1].set(title="log(Price) - close to normal", xlabel="log(Price)", ylabel="Count")
fig.text(0.5, -0.03, f"skewness: raw = {df['price'].skew():.2f}, log = {np.log(df['price']).skew():.2f}",
         ha="center", color=INK2)
save(fig, "01_price_distribution.png")

# correlation heatmap
corr = df[["price"] + FEATURES_RAW].corr()
fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
ax.grid(False)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(v) > 0.6 else INK)
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
ax.set_title("Correlation matrix (price + 18 features)")
save(fig, "02_correlation_heatmap.png")

# top features vs price
top = corr["price"].drop("price").abs().sort_values(ascending=False).head(6).index
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
samp = df.sample(4000, random_state=SEED)
for a, f in zip(axes.ravel(), top):
    a.scatter(samp[f], samp["price"] / 1e6, s=6, alpha=0.35, color=BLUE, linewidths=0)
    a.set(title=f"{f}  (r = {corr.loc[f, 'price']:.2f})", xlabel=f, ylabel="Price (M USD)")
fig.suptitle("Top-6 features most correlated with price (4,000-row sample)", fontweight="bold")
fig.tight_layout()
save(fig, "03_top_features_scatter.png")

# =====================================================================
# 3. Data Preparation
# =====================================================================
print("\n" + "=" * 70)
print("3. DATA PREPARATION")
print("=" * 70)
d = df.copy()
d["date"] = pd.to_datetime(d["date"].str[:8], format="%Y%m%d")
n0 = len(d)
d = d.sort_values("date").drop_duplicates("id", keep="last")      # keep latest sale per house
d = d[(d["bedrooms"] > 0) & (d["bedrooms"] < 15) & (d["bathrooms"] > 0)]  # data errors (e.g. 33 bedrooms)
print(f"rows: {n0} -> {len(d)} (removed duplicates sales & impossible records)")

# feature engineering
d["log_price"] = np.log(d["price"])
d["sale_year"] = d["date"].dt.year
d["house_age"] = d["sale_year"] - d["yr_built"]
d["is_renovated"] = (d["yr_renovated"] > 0).astype(int)
d["has_basement"] = (d["sqft_basement"] > 0).astype(int)
for c in ["sqft_living", "sqft_lot", "sqft_above", "sqft_living15", "sqft_lot15"]:
    d[f"log_{c}"] = np.log(d[c])
print("engineered: log_price, house_age, is_renovated, has_basement, log of area columns")
print("zipcode is nominal -> excluded from numeric MLR, used later as one-hot (location effect)")

CANDIDATES = ["bedrooms", "bathrooms", "log_sqft_living", "log_sqft_lot", "floors",
              "waterfront", "view", "condition", "grade", "log_sqft_above",
              "has_basement", "house_age", "is_renovated", "lat", "long",
              "log_sqft_living15", "log_sqft_lot15"]
TARGET = "log_price"

train, test = train_test_split(d, test_size=0.2, random_state=SEED)
print(f"train = {len(train)}, test = {len(test)} (80/20 split, seed={SEED})")
Xtr, ytr = train[CANDIDATES], train[TARGET]
Xte, yte = test[CANDIDATES], test[TARGET]

# =====================================================================
# 4. Modeling  (Feature Selection -> Models)
# =====================================================================
print("\n" + "=" * 70)
print("4. MODELING - FEATURE SELECTION")
print("=" * 70)

# 4.1 VIF (multicollinearity)
def vif_table(X):
    Xc = sm.add_constant(X)
    return pd.Series([variance_inflation_factor(Xc.values, i + 1) for i in range(X.shape[1])],
                     index=X.columns).sort_values(ascending=False)

vif = vif_table(Xtr)
print("VIF (all candidates):\n" + vif.round(2).to_string())

# 4.2 Backward elimination by p-value (statsmodels OLS)
def backward_elimination(X, y, alpha=0.05):
    cols, log = list(X.columns), []
    while True:
        m = sm.OLS(y, sm.add_constant(X[cols])).fit()
        p = m.pvalues.drop("const")
        if p.max() <= alpha:
            return cols, log
        worst = p.idxmax()
        log.append((worst, float(p.max())))
        cols.remove(worst)

be_cols, be_log = backward_elimination(Xtr, ytr)
print(f"Backward elimination removed: {be_log or 'none'}")

# 4.3 RFECV (recursive feature elimination with 5-fold CV)
kf = KFold(5, shuffle=True, random_state=SEED)
scaler = StandardScaler().fit(Xtr)
rfecv = RFECV(LinearRegression(), step=1, cv=kf, scoring="r2").fit(scaler.transform(Xtr), ytr)
rfe_rank = pd.Series(rfecv.ranking_, index=CANDIDATES).sort_values()
rfe_cols = list(np.array(CANDIDATES)[rfecv.support_])
cv_scores = rfecv.cv_results_["mean_test_score"]
print(f"RFECV optimal #features = {rfecv.n_features_}")

# 4.4 LassoCV (L1 shrinks weak features to 0)
lasso = LassoCV(cv=kf, random_state=SEED, n_alphas=100).fit(scaler.transform(Xtr), ytr)
lasso_coef = pd.Series(lasso.coef_, index=CANDIDATES)
print(f"LassoCV alpha = {lasso.alpha_:.5f}, non-zero = {(lasso_coef != 0).sum()}")

# 4.5 Final subset: start from features with |r| info + selection consensus, then
#     drop highly collinear duplicates (VIF > 5) keeping the stronger one.
votes = pd.DataFrame({
    "|r| with log_price": train[CANDIDATES].corrwith(train[TARGET]).abs(),
    "VIF": vif,
    "BackwardElim": [c in be_cols for c in CANDIDATES],
    "RFECV_rank": rfe_rank,
    "Lasso_coef(std)": lasso_coef,
}).loc[CANDIDATES]
votes["votes"] = (votes["BackwardElim"].astype(int)
                  + (votes["RFECV_rank"] == 1).astype(int)
                  + (votes["Lasso_coef(std)"].abs() > 0.01).astype(int))
selected = [c for c in CANDIDATES if votes.loc[c, "votes"] >= 2]

# iterative VIF pruning on the selected set
while True:
    v = vif_table(Xtr[selected])
    if v.max() <= 5:
        break
    # take the highest-VIF feature and its most-correlated partner,
    # drop whichever of the pair is weaker in |r| with target
    worst = v.idxmax()
    partner = Xtr[selected].corr()[worst].drop(worst).abs().idxmax()
    pair = votes.loc[[worst, partner], "|r| with log_price"]
    selected.remove(pair.idxmin())
votes["selected"] = [c in selected for c in CANDIDATES]
votes.round(4).to_csv(os.path.join(RES, "feature_selection.csv"))
print("\nFeature selection summary:\n" + votes.round(3).to_string())
print(f"\nFINAL SELECTED ({len(selected)}): {selected}")

# fig: RFECV curve + Lasso coefficients
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
xs = np.arange(1, len(cv_scores) + 1)
ax[0].plot(xs, cv_scores, color=BLUE, lw=2, marker="o", ms=5)
ax[0].axvline(rfecv.n_features_, color=GRAY, ls="--", lw=1)
ax[0].annotate(f"optimum = {rfecv.n_features_}", (rfecv.n_features_, cv_scores[rfecv.n_features_ - 1]),
               xytext=(-90, -30), textcoords="offset points", color=INK2,
               arrowprops=dict(arrowstyle="-", color=GRAY))
ax[0].set(title="RFECV: 5-fold CV R² vs. number of features", xlabel="# features", ylabel="CV R² (log price)")
lc = lasso_coef.reindex(lasso_coef.abs().sort_values().index)
cols_ = [BLUE if c in selected else GRAY for c in lc.index]
ax[1].barh(lc.index, lc.values, color=cols_, height=0.7)
ax[1].axvline(0, color=INK2, lw=0.8)
ax[1].set(title="LassoCV standardized coefficients", xlabel="coefficient (log price per 1 SD)")
ax[1].text(0.98, 0.04, "blue = in final subset\ngray = dropped", transform=ax[1].transAxes,
           ha="right", va="bottom", color=INK2, fontsize=9)
fig.tight_layout()
save(fig, "04_feature_selection.png")

# 4.6 Cross-check: forward stepwise selection by AIC (not used for voting, only to confirm
#     how many features are worth keeping and that the chosen subset is near the AIC/BIC elbow)
def forward_stepwise(X, y):
    chosen, rest, path = [], list(X.columns), []
    while rest:
        fits = {c: sm.OLS(y, sm.add_constant(X[chosen + [c]])).fit() for c in rest}
        best = min(fits, key=lambda c: fits[c].aic)
        chosen.append(best)
        rest.remove(best)
        m = fits[best]
        path.append({"step": len(chosen), "added": best, "AIC": m.aic, "BIC": m.bic, "AdjR2": m.rsquared_adj})
    return pd.DataFrame(path)

fwd = forward_stepwise(Xtr, ytr)
fwd.round(4).to_csv(os.path.join(RES, "forward_stepwise.csv"), index=False)
print("\nForward stepwise (AIC) order: " + " -> ".join(fwd["added"]))
print(f"BIC minimum at {int(fwd.loc[fwd['BIC'].idxmin(), 'step'])} features; "
      f"Adj R² at 12 features = {fwd.loc[11, 'AdjR2']:.4f} vs all 17 = {fwd['AdjR2'].iloc[-1]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(fwd["step"], fwd["AIC"] / 1000, color=BLUE, lw=2, marker="o", ms=4, label="AIC")
ax[0].plot(fwd["step"], fwd["BIC"] / 1000, color=ORANGE, lw=2, marker="s", ms=4, label="BIC")
ax[0].axvline(len(selected), color=GRAY, ls="--", lw=1)
ax[0].text(len(selected) + 0.2, 0.95, f"final subset = {len(selected)}", color=INK2, va="top", fontsize=9,
           transform=ax[0].get_xaxis_transform())
ax[0].set(title="Forward stepwise: AIC / BIC vs. # features", xlabel="# features", ylabel="information criterion (×1000)")
ax[0].legend()
ax[1].plot(fwd["step"], fwd["AdjR2"], color=BLUE, lw=2, marker="o", ms=4)
ax[1].set_xticks(fwd["step"], [f"+{c}" for c in fwd["added"]], rotation=60, ha="right", fontsize=8)
ax[1].axvline(len(selected), color=GRAY, ls="--", lw=1)
ax[1].set(title="Forward stepwise: Adjusted R² as features enter", ylabel="Adjusted R² (train, log price)")
fig.tight_layout()
save(fig, "12_forward_stepwise.png")

# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. MODELING - MODELS")
print("=" * 70)
results, cv_rows = {}, {}


def cv_r2(estimator, X, y):
    return cross_val_score(estimator, X, y, cv=kf, scoring="r2")


# M1 simple linear regression
m1_feat = ["log_sqft_living"]
m1 = sm.OLS(ytr, sm.add_constant(Xtr[m1_feat])).fit()
results["M1 Simple LR (log_sqft_living)"] = dollar_metrics(yte, m1.predict(sm.add_constant(Xte[m1_feat])), k=1)
cv_rows["M1 Simple LR (log_sqft_living)"] = cv_r2(LinearRegression(), Xtr[m1_feat], ytr)

# M2 MLR with all 17 candidates
m2 = sm.OLS(ytr, sm.add_constant(Xtr)).fit()
results[f"M2 MLR all ({len(CANDIDATES)} feat.)"] = dollar_metrics(yte, m2.predict(sm.add_constant(Xte)), k=len(CANDIDATES))
cv_rows[f"M2 MLR all ({len(CANDIDATES)} feat.)"] = cv_r2(LinearRegression(), Xtr, ytr)

# M3 MLR with selected features  <-- main interpretable model
m3 = sm.OLS(ytr, sm.add_constant(Xtr[selected])).fit()
M3_NAME = f"M3 MLR selected ({len(selected)} feat.)"
results[M3_NAME] = dollar_metrics(yte, m3.predict(sm.add_constant(Xte[selected])), k=len(selected))
cv_rows[M3_NAME] = cv_r2(LinearRegression(), Xtr[selected], ytr)
with open(os.path.join(RES, "m3_ols_summary.txt"), "w", encoding="utf-8") as f:
    f.write(m3.summary().as_text())

# M3 assumption tests: heteroscedasticity, normality, autocorrelation, and a robust-SE check
bp_lm, bp_p, _, _ = het_breuschpagan(m3.resid, m3.model.exog)
jb_stat, jb_p, skew_, kurt_ = jarque_bera(m3.resid)
m3_hc3 = m3.get_robustcov_results(cov_type="HC3")
hc3_p = pd.Series(m3_hc3.pvalues, index=m3.params.index).drop("const")
diagnostics = {
    "Breusch-Pagan LM": float(bp_lm), "Breusch-Pagan p": float(bp_p),
    "Jarque-Bera": float(jb_stat), "Jarque-Bera p": float(jb_p),
    "residual skew": float(skew_), "residual kurtosis": float(kurt_),
    "Durbin-Watson": float(durbin_watson(m3.resid)),
    "condition number (std. X)": float(np.linalg.cond(sm.add_constant(
        (Xtr[selected] - Xtr[selected].mean()) / Xtr[selected].std()).values)),
    "max VIF": float(vif_table(Xtr[selected]).max()),
    "features significant with HC3 robust SE (p<0.05)": f"{int((hc3_p < 0.05).sum())}/{len(hc3_p)}",
    "not significant with HC3": ", ".join(f"{c} (p={hc3_p[c]:.3f})" for c in hc3_p.index[hc3_p >= 0.05]) or "none",
}
print("\nM3 assumption checks:")
for k, v in diagnostics.items():
    print(f"  {k:<48s} {v if isinstance(v, str) else round(v, 4)}")

# M4 MLR selected + zipcode one-hot (location fixed effect) <-- best linear model
zip_levels = sorted(d["zipcode"].unique())


def with_zip(frame):
    z = pd.get_dummies(pd.Categorical(frame["zipcode"], categories=zip_levels), prefix="zip", drop_first=True).astype(float)
    z.index = frame.index
    return pd.concat([frame[selected], z], axis=1)


Xtr4, Xte4 = with_zip(train), with_zip(test)
m4 = sm.OLS(ytr, sm.add_constant(Xtr4)).fit()
M4_NAME = "M4 MLR selected + zipcode one-hot"
results[M4_NAME] = dollar_metrics(yte, m4.predict(sm.add_constant(Xte4)), k=Xtr4.shape[1])
cv_rows[M4_NAME] = cv_r2(LinearRegression(), Xtr4, ytr)

# Regularized linear models (same design as M4)
ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))).fit(Xtr4, ytr)
results["Ridge (M4 design)"] = dollar_metrics(yte, ridge.predict(Xte4))
cv_rows["Ridge (M4 design)"] = cv_r2(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 30))), Xtr4, ytr)

# Mainstream non-linear benchmarks seen on Kaggle (tree ensembles)
XtrT = train[CANDIDATES + ["zipcode"]]
XteT = test[CANDIDATES + ["zipcode"]]
rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED, min_samples_leaf=2).fit(XtrT, ytr)
results["Random Forest (benchmark)"] = dollar_metrics(yte, rf.predict(XteT))
cv_rows["Random Forest (benchmark)"] = cv_r2(
    RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED, min_samples_leaf=2), XtrT, ytr)
hgb = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, random_state=SEED).fit(XtrT, ytr)
results["HistGradientBoosting (benchmark)"] = dollar_metrics(yte, hgb.predict(XteT))
cv_rows["HistGradientBoosting (benchmark)"] = cv_r2(
    HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, random_state=SEED), XtrT, ytr)

# =====================================================================
# 5. Evaluation
# =====================================================================
print("\n" + "=" * 70)
print("5. EVALUATION")
print("=" * 70)
res = pd.DataFrame(results).T
res["CV_R2_log (5-fold mean)"] = pd.Series({k: v.mean() for k, v in cv_rows.items()})
res["CV_R2_log (std)"] = pd.Series({k: v.std() for k, v in cv_rows.items()})
res = res[["R2_log", "AdjR2_log", "CV_R2_log (5-fold mean)", "CV_R2_log (std)", "R2_$", "RMSE_$", "MAE_$", "MAPE_%"]]
res.to_csv(os.path.join(RES, "model_comparison.csv"))
pd.set_option("display.width", 200)
print(res.round(4).to_string())

# 95% CI / PI for M3 and M4 on test set (statsmodels)
def interval_frame(model, Xte_design):
    sf = model.get_prediction(sm.add_constant(Xte_design, has_constant="add")).summary_frame(alpha=0.05)
    sf.index = Xte_design.index
    return sf

sf3 = interval_frame(m3, Xte[selected])
sf4 = interval_frame(m4, Xte4)
coverage = {}
for name, sf in [(M3_NAME, sf3), (M4_NAME, sf4)]:
    inside = (yte >= sf["obs_ci_lower"]) & (yte <= sf["obs_ci_upper"])
    width = np.exp(sf["obs_ci_upper"]) - np.exp(sf["obs_ci_lower"])
    coverage[name] = {"PI95_coverage_%": float(inside.mean() * 100),
                      "median_PI_width_$": float(width.median())}
    print(f"{name}: 95% PI coverage on test = {inside.mean()*100:.2f}%, median width = ${width.median():,.0f}")

# PI calibration: nominal level vs. empirical coverage on the test set
LEVELS = [0.50, 0.68, 0.80, 0.90, 0.95, 0.99]
calib = {}
for name, model, Xd in [(M3_NAME, m3, Xte[selected]), (M4_NAME, m4, Xte4)]:
    row = {}
    for lv in LEVELS:
        s = model.get_prediction(sm.add_constant(Xd, has_constant="add")).summary_frame(alpha=1 - lv)
        row[f"{int(lv*100)}%"] = float(((yte.values >= s["obs_ci_lower"].values) &
                                        (yte.values <= s["obs_ci_upper"].values)).mean() * 100)
    calib[name] = row
calib_df = pd.DataFrame(calib).T
calib_df.round(2).to_csv(os.path.join(RES, "pi_calibration.csv"))
print("\nPI calibration (nominal -> empirical coverage %):\n" + calib_df.round(1).to_string())

with open(os.path.join(RES, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump({"selected_features": selected,
               "forward_stepwise_order": list(fwd["added"]),
               "m3_diagnostics": diagnostics,
               "pi_calibration_%": calib,
               "rfecv_n_features": int(rfecv.n_features_),
               "lasso_alpha": float(lasso.alpha_),
               "backward_elim_removed": be_log,
               "models": res.round(6).to_dict(orient="index"),
               "interval_coverage": coverage,
               "m3_coefficients": m3.params.round(6).to_dict(),
               "m3_pvalues": m3.pvalues.round(6).to_dict(),
               "rows_after_cleaning": int(len(d)),
               "train_rows": int(len(train)), "test_rows": int(len(test))}, f, indent=2)

# ---- Fig 5: simple LR with CI and PI bands ----
grid = pd.DataFrame({"log_sqft_living": np.linspace(Xte["log_sqft_living"].min(), Xte["log_sqft_living"].max(), 200)})
g = m1.get_prediction(sm.add_constant(grid)).summary_frame(alpha=0.05)
sx = np.exp(grid["log_sqft_living"])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ts = test.sample(2500, random_state=SEED)
for a, logscale in zip(ax, [True, False]):
    tr = (lambda v: v) if logscale else (lambda v: np.exp(v) / 1e6)
    a.scatter(ts["sqft_living"], tr(ts["log_price"]), s=6, alpha=0.3, color=GRAY, linewidths=0, label="test data")
    a.fill_between(sx, tr(g["obs_ci_lower"]), tr(g["obs_ci_upper"]), color=AQUA, alpha=0.18, label="95% prediction interval")
    a.fill_between(sx, tr(g["mean_ci_lower"]), tr(g["mean_ci_upper"]), color=ORANGE, alpha=0.5, label="95% confidence interval")
    a.plot(sx, tr(g["mean"]), color=BLUE, lw=2, label="fitted line")
    a.set_xscale("log" if logscale else "linear")
    a.set_xlabel("sqft_living" + (" (log scale)" if logscale else ""))
    a.set_ylabel("log(Price)" if logscale else "Price (million USD)")
ax[0].set_title("M1 simple LR: log(price) ~ log(sqft_living)")
ax[1].set_title("Same model back-transformed to USD")
ax[1].set_ylim(0, 3)
ax[0].legend(loc="upper left", fontsize=9)
fig.tight_layout()
save(fig, "05_simple_lr_ci_pi.png")

# ---- Fig 6: MLR prediction plot with PI (sorted test samples) ----
def pi_plot(sf, name, fname, n=120):
    sub = sf.sample(n, random_state=SEED).sort_values("mean")
    x = np.arange(n)
    act = np.exp(yte.loc[sub.index]) / 1e6
    lo, hi = np.exp(sub["obs_ci_lower"]) / 1e6, np.exp(sub["obs_ci_upper"]) / 1e6
    clo, chi = np.exp(sub["mean_ci_lower"]) / 1e6, np.exp(sub["mean_ci_upper"]) / 1e6
    inside = (act >= lo) & (act <= hi)
    fig, a = plt.subplots(figsize=(13, 4.8))
    a.fill_between(x, lo, hi, color=AQUA, alpha=0.2, step="mid", label="95% prediction interval")
    a.fill_between(x, clo, chi, color=ORANGE, alpha=0.55, step="mid", label="95% confidence interval")
    a.plot(x, np.exp(sub["mean"]) / 1e6, color=BLUE, lw=2, label="predicted price")
    a.scatter(x[inside], act[inside], s=22, color=INK, zorder=3, label="actual (inside PI)",
              edgecolors="white", linewidths=0.8)
    a.scatter(x[~inside], act[~inside], s=40, marker="x", color=ORANGE, zorder=3, label="actual (outside PI)")
    a.set(title=f"{name}: test-set predictions with 95% CI / PI ({n} random houses, sorted)",
          xlabel="test houses (sorted by predicted price)", ylabel="Price (million USD)")
    a.legend(ncol=5, loc="upper left", fontsize=9)
    save(fig, fname)

pi_plot(sf3, M3_NAME, "06_mlr_selected_prediction_interval.png")
pi_plot(sf4, M4_NAME, "07_mlr_zip_prediction_interval.png")

# ---- Fig 8: actual vs predicted ----
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for a, (nm, sf) in zip(ax, [(M3_NAME, sf3), (M4_NAME, sf4)]):
    p = np.exp(sf["mean"]) / 1e6
    t = np.exp(yte) / 1e6
    a.scatter(t, p, s=6, alpha=0.3, color=BLUE, linewidths=0)
    lim = [0, 3]
    a.plot(lim, lim, color=INK2, lw=1, ls="--")
    a.set(xlim=lim, ylim=lim, xlabel="Actual price (M USD)", ylabel="Predicted price (M USD)",
          title=f"{nm}\nR²($) = {results[nm]['R2_$']:.3f}, MAPE = {results[nm]['MAPE_%']:.1f}%")
fig.tight_layout()
save(fig, "08_actual_vs_predicted.png")

# ---- Fig 9: residual diagnostics of M3 ----
resid = m3.resid
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(m3.fittedvalues, resid, s=5, alpha=0.25, color=BLUE, linewidths=0)
ax[0].axhline(0, color=INK2, lw=1)
ax[0].set(title="Residuals vs fitted", xlabel="fitted log(price)", ylabel="residual")
ax[1].hist(resid, bins=70, color=BLUE, edgecolor="white", linewidth=0.4)
ax[1].set(title="Residual distribution", xlabel="residual")
(osm, osr), (slope, inter, _) = stats.probplot(resid, dist="norm")
ax[2].scatter(osm, osr, s=5, color=BLUE, alpha=0.4, linewidths=0)
ax[2].plot(osm, slope * osm + inter, color=ORANGE, lw=2)
ax[2].set(title="Normal Q-Q", xlabel="theoretical quantiles", ylabel="sample quantiles")
fig.suptitle(f"{M3_NAME} residual diagnostics (train)", fontweight="bold")
fig.tight_layout()
save(fig, "09_residual_diagnostics.png")

# ---- Fig 10: model comparison ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
order = res.index[::-1]
is_lin = ["benchmark" not in n for n in order]
ax[0].barh(order, res.loc[order, "R2_log"], color=[BLUE if l else GRAY for l in is_lin], height=0.65)
for i, v in enumerate(res.loc[order, "R2_log"]):
    ax[0].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
ax[0].set(title="Test R² (log price) - higher is better", xlim=(0, 1.02))
ax[1].barh(order, res.loc[order, "MAPE_%"], color=[BLUE if l else GRAY for l in is_lin], height=0.65)
for i, v in enumerate(res.loc[order, "MAPE_%"]):
    ax[1].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9, color=INK)
ax[1].set(title="Test MAPE (USD) - lower is better")
ax[1].set_yticklabels([])
fig.text(0.99, -0.02, "blue = linear regression family · gray = tree-ensemble benchmarks", ha="right", color=INK2)
fig.tight_layout()
save(fig, "10_model_comparison.png")

# ---- Fig 11: M3 coefficients with 95% CI ----
ci = m3.conf_int().drop("const")
coef = m3.params.drop("const")
std = Xtr[selected].std()
sc = (coef * std).sort_values()
sci = ci.loc[sc.index].mul(std.loc[sc.index], axis=0)
fig, a = plt.subplots(figsize=(8, 0.45 * len(sc) + 1.5))
a.errorbar(sc.values, range(len(sc)), xerr=[sc - sci[0], sci[1] - sc], fmt="o", color=BLUE, ms=6,
           ecolor=BLUE, elinewidth=2, capsize=0)
a.axvline(0, color=INK2, lw=0.8)
a.set_yticks(range(len(sc)), sc.index)
a.set(title=f"{M3_NAME}: standardized coefficients ± 95% CI",
      xlabel="effect on log(price) of +1 SD  (≈ % change ×100)")
save(fig, "11_coefficients_ci.png")

# ---- Fig 13: PI calibration ----
fig, a = plt.subplots(figsize=(6.2, 5))
nom = [lv * 100 for lv in LEVELS]
a.plot([45, 100], [45, 100], color=INK2, ls="--", lw=1, label="perfect calibration")
for (name, row), col, mk in zip(calib.items(), [BLUE, ORANGE], ["o", "s"]):
    a.plot(nom, list(row.values()), color=col, lw=2, marker=mk, ms=6, label=name)
for x_, y_ in zip(nom, calib[M4_NAME].values()):
    a.annotate(f"{y_:.1f}", (x_, y_), xytext=(-12, 9), textcoords="offset points", fontsize=8, color=ORANGE)
a.set(xlim=(45, 100), ylim=(45, 100), xlabel="nominal prediction-interval level (%)",
      ylabel="empirical coverage on test set (%)", title="Prediction-interval calibration (4,284 test houses)")
a.legend(loc="upper left", fontsize=8.5)
save(fig, "13_pi_calibration.png")

# ---- Fig 14: residual map (why zipcode helps) ----
r3 = yte - sf3["mean"]
r4 = yte - sf4["mean"]
fig, ax = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True, layout="constrained")
for a, (nm, r) in zip(ax, [(M3_NAME, r3), (M4_NAME, r4)]):
    sc_ = a.scatter(test["long"], test["lat"], c=r.clip(-0.6, 0.6), cmap="RdBu_r", vmin=-0.6, vmax=0.6,
                    s=7, alpha=0.8, linewidths=0)
    a.set(title=f"{nm}\nresidual SD = {r.std():.3f}", xlabel="longitude")
    a.set_aspect(1 / np.cos(np.radians(47.5)))
    a.grid(False)
ax[0].set_ylabel("latitude")
fig.colorbar(sc_, ax=ax, shrink=0.85, label="residual log(price)  (red = under-predicted)")
fig.suptitle("Test-set residuals on the map: location clusters disappear after adding zipcode",
             fontweight="bold")
save(fig, "14_residual_map.png")

# =====================================================================
# 5b. Supplement: Auto Regression on the weekly median price
#     (the assignment allows AR; the data are cross-sectional, so AR can only model
#      the market-level time trend, not individual houses)
# =====================================================================
print("\n" + "=" * 70)
print("5b. SUPPLEMENT - AUTO REGRESSION (weekly median price)")
print("=" * 70)
wk = d.set_index("date")["price"].resample("W").median().dropna()
wk = wk.iloc[1:-1]  # drop the partial first/last week
ylog = np.log(wk.values)
H = 8  # hold out the last 8 weeks
sel = ar_select_order(ylog[:-H], maxlag=6, ic="aic", trend="c")
p_ar = max(sel.ar_lags) if sel.ar_lags else 1
ar = AutoReg(ylog[:-H], lags=p_ar, trend="c").fit()
fc = ar.get_prediction(start=len(ylog) - H, end=len(ylog) - 1)
fc_mean, fc_ci = fc.predicted_mean, fc.conf_int(alpha=0.05)
naive = np.repeat(ylog[-H - 1], H)
ar_metrics = {
    "weeks": int(len(wk)), "lag_order_AIC": int(p_ar), "holdout_weeks": H,
    "AR_MAPE_%": float(np.mean(np.abs(np.exp(fc_mean) - wk.values[-H:]) / wk.values[-H:]) * 100),
    "naive_MAPE_%": float(np.mean(np.abs(np.exp(naive) - wk.values[-H:]) / wk.values[-H:]) * 100),
    "PI95_coverage_%": float(((ylog[-H:] >= fc_ci[:, 0]) & (ylog[-H:] <= fc_ci[:, 1])).mean() * 100),
}
print(f"  {len(wk)} weekly medians, AR({p_ar}) chosen by AIC")
print(f"  last {H} weeks: AR MAPE = {ar_metrics['AR_MAPE_%']:.2f}%, naive (last value) MAPE = "
      f"{ar_metrics['naive_MAPE_%']:.2f}%, 95% PI coverage = {ar_metrics['PI95_coverage_%']:.0f}%")
with open(os.path.join(RES, "autoregression.json"), "w", encoding="utf-8") as f:
    json.dump(ar_metrics, f, indent=2)

fig, a = plt.subplots(figsize=(12, 4.2))
a.plot(wk.index, wk.values / 1e3, color=GRAY, lw=1.5, marker="o", ms=3, label="weekly median price")
a.plot(wk.index[:-H], np.exp(np.r_[np.full(p_ar, np.nan), ar.fittedvalues]) / 1e3, color=BLUE, lw=1.5,
       label=f"AR({p_ar}) in-sample fit")
a.fill_between(wk.index[-H:], np.exp(fc_ci[:, 0]) / 1e3, np.exp(fc_ci[:, 1]) / 1e3, color=AQUA, alpha=0.25,
               label="95% forecast interval")
a.plot(wk.index[-H:], np.exp(fc_mean) / 1e3, color=ORANGE, lw=2, marker="s", ms=4, label=f"AR forecast ({H} weeks)")
a.axvline(wk.index[-H], color=INK2, ls="--", lw=0.8)
a.set(title=f"Supplement: AR({p_ar}) on weekly median sale price - forecast MAPE {ar_metrics['AR_MAPE_%']:.1f}% "
            f"(naive {ar_metrics['naive_MAPE_%']:.1f}%)", ylabel="median price (thousand USD)")
a.legend(ncol=4, loc="upper left", fontsize=8.5)
save(fig, "15_autoregression_weekly.png")

# =====================================================================
# 6. Deployment
# =====================================================================
print("\n" + "=" * 70)
print("6. DEPLOYMENT")
print("=" * 70)
m4.remove_data()  # drop stored training arrays (34 MB -> small); predictions/intervals still work
bundle = {"model": m4, "features": selected, "zip_levels": zip_levels,
          "note": "statsmodels OLS on log(price); use predict_price() in predict.py"}
with open(os.path.join(MOD, "house_price_mlr.pkl"), "wb") as f:
    pickle.dump(bundle, f)
print("  saved model/house_price_mlr.pkl  (M4: selected features + zipcode)")

# demo prediction with interval for one test house
demo = test.iloc[[0]]
sfd = m4.get_prediction(sm.add_constant(with_zip(demo), has_constant="add")).summary_frame(alpha=0.05)
print(f"  demo house id={demo['id'].iloc[0]}: actual ${demo['price'].iloc[0]:,.0f}, "
      f"predicted ${np.exp(sfd['mean'].iloc[0]):,.0f} "
      f"(95% PI ${np.exp(sfd['obs_ci_lower'].iloc[0]):,.0f} - ${np.exp(sfd['obs_ci_upper'].iloc[0]):,.0f})")
print("\nDone.")
