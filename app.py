"""
HW2 deployment website: King County house price estimator (Streamlit).
Run 7115064191_hw2.py first to create model/house_price_mlr.pkl.gz, then:
    streamlit run app.py
"""
import gzip
import json
import os
import pickle

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from predict import explain_price, predict_price

BASE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="King County 房價估算", page_icon="🏠", layout="wide")


@st.cache_resource
def load_model():
    with gzip.open(os.path.join(BASE, "model", "house_price_mlr.pkl.gz"), "rb") as f:
        return pickle.load(f)


@st.cache_data
def zip_locations():
    df = pd.read_csv(os.path.join(BASE, "kc_house_data.csv"), usecols=["zipcode", "lat", "long", "price"])
    return df.groupby("zipcode").agg(lat=("lat", "median"), long=("long", "median"),
                                     median_price=("price", "median"), sales=("price", "size"))


@st.cache_data
def model_metrics():
    with open(os.path.join(BASE, "results", "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    name = "M4 MLR selected + zipcode one-hot"
    return m["models"][name], m["interval_coverage"][name]


bundle = load_model()
zips = zip_locations()
m4, cov = model_metrics()

st.title("🏠 King County 房價估算")
st.caption("HW2 多元線性回歸部署｜學號 7115064191｜模型 M4：12 個特徵 + zipcode 地段效果，"
           "以 2014–2015 年 21,419 筆成交資料訓練（Kaggle: House Sales in King County, USA）")

# ---------------- inputs ----------------
with st.sidebar:
    st.header("房屋條件")
    zipcode = st.selectbox("郵遞區號 zipcode", list(zips.index), index=list(zips.index).index(98052),
                           format_func=lambda z: f"{z}（成交中位數 ${zips.loc[z, 'median_price'] / 1e3:,.0f}K）")
    sqft_living = st.slider("室內面積 sqft_living（平方英尺）", 400, 8000, 2000, step=50)
    sqft_lot = st.number_input("土地面積 sqft_lot（平方英尺）", 500, 200000, 6000, step=500)
    bedrooms = st.slider("臥室數", 1, 8, 3)
    floors = st.select_slider("樓層數", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], value=1.0)
    grade = st.slider("建築等級 grade（1–13）", 3, 13, 8)
    condition = st.slider("屋況 condition（1–5）", 1, 5, 3)
    view = st.slider("景觀 view（0–4）", 0, 4, 0)
    waterfront = st.checkbox("臨水 waterfront")
    sqft_basement = st.number_input("地下室面積（0 = 沒有）", 0, 3000, 0, step=100)
    yr_built = st.slider("建造年份", 1900, 2015, 1990)
    renovated = st.checkbox("曾經翻修")

house = dict(bedrooms=bedrooms, sqft_living=sqft_living, sqft_lot=sqft_lot, floors=floors,
             waterfront=int(waterfront), view=view, condition=condition, grade=grade,
             sqft_basement=sqft_basement, yr_built=yr_built, yr_renovated=2010 if renovated else 0,
             zipcode=int(zipcode), lat=float(zips.loc[zipcode, "lat"]), sale_year=2015)

out = predict_price(house, bundle)
base, eff = explain_price(house, bundle)

# ---------------- results ----------------
c1, c2, c3 = st.columns(3)
c1.metric("預測成交價", f"${out['predicted_price']:,.0f}",
          f"{(out['predicted_price'] / base - 1) * 100:+.0f}% vs 一般房子", delta_color="off")
D = "\\$"  # escaped dollar: two "$" in one string would be rendered as LaTeX math
c2.metric("95% 預測區間（單一房屋）", f"{D}{out['PI95'][0] / 1e3:,.0f}K – {D}{out['PI95'][1] / 1e3:,.0f}K")
c3.metric("95% 信賴區間（同條件平均價）", f"{D}{out['CI95'][0] / 1e3:,.0f}K – {D}{out['CI95'][1] / 1e3:,.0f}K")

left, right = st.columns([3, 2])
with left:
    st.subheader("為什麼是這個價格？")
    st.caption(f"從「一般房子」${base:,.0f}（所有特徵都在訓練集平均值）出發，每個條件讓價格變動的百分比；"
               "全部相乘剛好等於預測價。")
    names = {"log_sqft_living": "室內面積", "log_sqft_lot": "土地面積", "grade": "建築等級", "condition": "屋況",
             "view": "景觀", "waterfront": "臨水", "bedrooms": "臥室數", "floors": "樓層數",
             "has_basement": "地下室", "house_age": "屋齡", "is_renovated": "翻修", "lat": "緯度"}
    chart_df = pd.DataFrame({
        "條件": [names.get(k, k.replace("zipcode", "地段 zipcode")) for k in eff.index],
        "影響 (%)": np.expm1(eff.values) * 100,
    })
    chart_df["方向"] = np.where(chart_df["影響 (%)"] >= 0, "提高", "降低")
    bars = alt.Chart(chart_df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("影響 (%):Q", title="對價格的影響（%）",
                scale=alt.Scale(domain=[min(0, chart_df["影響 (%)"].min()) * 1.25 - 1,
                                        max(0, chart_df["影響 (%)"].max()) * 1.25 + 1])),
        y=alt.Y("條件:N", sort=None, title=None),
        color=alt.Color("方向:N", scale=alt.Scale(domain=["提高", "降低"], range=["#2a78d6", "#eb6834"]), legend=None),
        tooltip=["條件", alt.Tooltip("影響 (%):Q", format="+.1f")],
    )
    pos = bars.transform_filter(alt.datum["影響 (%)"] >= 0).mark_text(align="left", dx=4, color="#52514e").encode(
        text=alt.Text("影響 (%):Q", format="+.1f"))
    neg = bars.transform_filter(alt.datum["影響 (%)"] < 0).mark_text(align="right", dx=-4, color="#52514e").encode(
        text=alt.Text("影響 (%):Q", format="+.1f"))
    st.altair_chart((bars + pos + neg).properties(height=360), use_container_width=True)

with right:
    st.subheader("位置")
    zdf = zips.reset_index()
    zdf["中位價 (K)"] = zdf["median_price"] / 1e3
    allz = alt.Chart(zdf).mark_circle(size=90, opacity=0.85).encode(
        x=alt.X("long:Q", scale=alt.Scale(zero=False), title="經度"),
        y=alt.Y("lat:Q", scale=alt.Scale(zero=False), title="緯度"),
        color=alt.Color("中位價 (K):Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(title="中位價 $K")),
        tooltip=["zipcode", alt.Tooltip("中位價 (K):Q", format=",.0f"), "sales"],
    )
    here = alt.Chart(zdf[zdf["zipcode"] == zipcode]).mark_point(size=420, shape="diamond", filled=True,
                                                               color="#eb6834").encode(x="long:Q", y="lat:Q")
    st.altair_chart((allz + here).properties(height=360), use_container_width=True)
    st.caption(f"每個點是一個 zipcode（顏色＝成交中位價），◆ 為目前選擇。zipcode {zipcode}：資料集中 {int(zips.loc[zipcode, 'sales'])} 筆成交，"
               f"中位數 ${zips.loc[zipcode, 'median_price']:,.0f}")

st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("測試集 R²（log）", f"{m4['R2_log']:.3f}")
k2.metric("5-fold CV R²", f"{m4['CV_R2_log (5-fold mean)']:.3f}")
k3.metric("MAPE", f"{m4['MAPE_%']:.1f}%")
k4.metric("95% 預測區間實際覆蓋率", f"{cov['PI95_coverage_%']:.1f}%")
st.caption("模型評估來自 4,284 間測試房屋。資料只涵蓋 2014/05–2015/05，實際估價前應以最新成交資料重新訓練。")
