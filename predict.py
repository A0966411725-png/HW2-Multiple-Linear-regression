"""
Deployment demo: load the saved MLR model and predict a house price with a 95% interval.
Run 7115064191_hw2.py first to create model/house_price_mlr.pkl.

Usage:
    python3 predict.py
    python3 predict.py --sqft_living 2500 --grade 9 --zipcode 98004 --lat 47.62
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.abspath(__file__))


def predict_price(house, bundle, alpha=0.05):
    """house: dict with raw columns (sqft_living, grade, zipcode, ...). Returns $ estimate and intervals."""
    h = pd.DataFrame([house])
    h["log_sqft_living"] = np.log(h["sqft_living"])
    h["log_sqft_lot"] = np.log(h["sqft_lot"])
    h["house_age"] = h["sale_year"] - h["yr_built"]
    h["is_renovated"] = (h["yr_renovated"] > 0).astype(int)
    h["has_basement"] = (h["sqft_basement"] > 0).astype(int)
    z = pd.get_dummies(pd.Categorical(h["zipcode"], categories=bundle["zip_levels"]),
                       prefix="zip", drop_first=True).astype(float)
    X = pd.concat([h[bundle["features"]], z], axis=1)
    sf = bundle["model"].get_prediction(sm.add_constant(X, has_constant="add")).summary_frame(alpha=alpha)
    r = sf.iloc[0]
    return {
        "predicted_price": float(np.exp(r["mean"])),
        "CI95": (float(np.exp(r["mean_ci_lower"])), float(np.exp(r["mean_ci_upper"]))),
        "PI95": (float(np.exp(r["obs_ci_lower"])), float(np.exp(r["obs_ci_upper"]))),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    defaults = dict(bedrooms=3, sqft_living=2000, sqft_lot=6000, floors=1.0, waterfront=0, view=0,
                    condition=3, grade=8, sqft_basement=0, yr_built=1990, yr_renovated=0,
                    zipcode=98052, lat=47.68, sale_year=2015)
    for k, v in defaults.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    args = vars(p.parse_args())

    with open(os.path.join(BASE, "model", "house_price_mlr.pkl"), "rb") as f:
        bundle = pickle.load(f)
    out = predict_price(args, bundle)
    print("Input:", args)
    print(f"Predicted price : ${out['predicted_price']:,.0f}")
    print(f"95% CI (mean)   : ${out['CI95'][0]:,.0f} - ${out['CI95'][1]:,.0f}")
    print(f"95% PI (house)  : ${out['PI95'][0]:,.0f} - ${out['PI95'][1]:,.0f}")
