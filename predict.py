"""
Deployment demo: load the saved MLR model, predict a house price with a 95% interval,
and explain the prediction feature by feature.
Run 7115064191_hw2.py first to create model/house_price_mlr.pkl.gz (gzip-compressed pickle).

Usage:
    python3 predict.py
    python3 predict.py --sqft_living 2500 --grade 9 --zipcode 98004 --lat 47.62
    python3 predict.py --explain          # also show how each feature moves the price
"""
import argparse
import gzip
import os
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.abspath(__file__))


def design(house, bundle):
    """Turn raw house columns into the model's design row (selected features + zipcode dummies)."""
    h = pd.DataFrame([house])
    h["log_sqft_living"] = np.log(h["sqft_living"])
    h["log_sqft_lot"] = np.log(h["sqft_lot"])
    h["house_age"] = h["sale_year"] - h["yr_built"]
    h["is_renovated"] = (h["yr_renovated"] > 0).astype(int)
    h["has_basement"] = (h["sqft_basement"] > 0).astype(int)
    z = pd.get_dummies(pd.Categorical(h["zipcode"], categories=bundle["zip_levels"]),
                       prefix="zip", drop_first=True).astype(float)
    return pd.concat([h[bundle["features"]], z], axis=1)


def predict_price(house, bundle, alpha=0.05):
    """house: dict with raw columns (sqft_living, grade, zipcode, ...). Returns $ estimate and intervals."""
    X = design(house, bundle)
    sf = bundle["model"].get_prediction(sm.add_constant(X, has_constant="add")).summary_frame(alpha=alpha)
    r = sf.iloc[0]
    return {
        "predicted_price": float(np.exp(r["mean"])),
        "CI95": (float(np.exp(r["mean_ci_lower"])), float(np.exp(r["mean_ci_upper"]))),
        "PI95": (float(np.exp(r["obs_ci_lower"])), float(np.exp(r["obs_ci_upper"]))),
    }


def explain_price(house, bundle):
    """Exact decomposition of the log-linear prediction:
    price = typical-house price x product of (1 + effect of each feature).
    The typical house has every feature at its training mean and an average zipcode effect."""
    X = design(house, bundle).iloc[0]
    params = bundle["model"].params
    feats, mean = bundle["features"], bundle["x_mean"]
    base_log = params["const"] + sum(params[f] * mean[f] for f in feats) + bundle["zip_mean_effect"]
    effects = {f: params[f] * (X[f] - mean[f]) for f in feats}
    zip_cols = [c for c in X.index if c.startswith("zip_")]
    effects[f"zipcode {house['zipcode']}"] = float(sum(params[c] * X[c] for c in zip_cols)) - bundle["zip_mean_effect"]
    eff = pd.Series(effects)
    eff = eff.reindex(eff.abs().sort_values(ascending=False).index)
    return float(np.exp(base_log)), eff


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    defaults = dict(bedrooms=3, sqft_living=2000, sqft_lot=6000, floors=1.0, waterfront=0, view=0,
                    condition=3, grade=8, sqft_basement=0, yr_built=1990, yr_renovated=0,
                    zipcode=98052, lat=47.68, sale_year=2015)
    for k, v in defaults.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--explain", action="store_true", help="show each feature's effect on the price")
    args = vars(p.parse_args())
    explain = args.pop("explain")

    with gzip.open(os.path.join(BASE, "model", "house_price_mlr.pkl.gz"), "rb") as f:
        bundle = pickle.load(f)
    out = predict_price(args, bundle)
    print("Input:", args)
    print(f"Predicted price : ${out['predicted_price']:,.0f}")
    print(f"95% CI (mean)   : ${out['CI95'][0]:,.0f} - ${out['CI95'][1]:,.0f}")
    print(f"95% PI (house)  : ${out['PI95'][0]:,.0f} - ${out['PI95'][1]:,.0f}")
    if explain:
        base, eff = explain_price(args, bundle)
        print(f"\nWhy? Typical King County house (all features at the training average): ${base:,.0f}")
        running = base
        for name, e in eff.items():
            running *= np.exp(e)
            print(f"  {name:<18s} {np.expm1(e) * 100:+7.1f}%   -> ${running:,.0f}")
        print(f"  = predicted price ${running:,.0f}")
