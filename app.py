"""
HW2 deployment website (Streamlit). A dropdown switches between problems:
  1. King County house price (this assignment's M4 model)
  2. 50 Startups profit (course example dataset)
  3. Any dataset the user enters (CSV URL or upload) -> automatic CRISP-DM multiple linear regression
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

from generic_mlr import METHODS, Pipeline, guess_feature_columns
from predict import explain_price, predict_price

BASE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="HW2 多元線性回歸", page_icon="📈", layout="wide")


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



def page_king_county():
    bundle = load_model()
    zips = zip_locations()
    m4, cov = model_metrics()

    st.title("🏠 King County 房價估算")
    st.caption("HW2 多元線性回歸部署｜學號 7115064191｜模型 M4：12 個特徵 + zipcode 地段效果，"
               "以 2014–2015 年 21,419 筆成交資料訓練（Kaggle: House Sales in King County, USA）")

    # ---------------- inputs ----------------
    with st.sidebar:
        st.subheader("房屋條件")
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
        st.altair_chart((bars + pos + neg).properties(height=360), width="stretch")

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
        st.altair_chart((allz + here).properties(height=360), width="stretch")
        st.caption(f"每個點是一個 zipcode（顏色＝成交中位價），◆ 為目前選擇。zipcode {zipcode}：資料集中 {int(zips.loc[zipcode, 'sales'])} 筆成交，"
                   f"中位數 ${zips.loc[zipcode, 'median_price']:,.0f}")

    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("測試集 R²（log）", f"{m4['R2_log']:.3f}")
    k2.metric("5-fold CV R²", f"{m4['CV_R2_log (5-fold mean)']:.3f}")
    k3.metric("MAPE", f"{m4['MAPE_%']:.1f}%")
    k4.metric("95% 預測區間實際覆蓋率", f"{cov['PI95_coverage_%']:.1f}%")
    st.caption("模型評估來自 4,284 間測試房屋。資料只涵蓋 2014/05–2015/05，實際估價前應以最新成交資料重新訓練。")

# =====================================================================
# Generic dataset pages (problem 2 and "any dataset")
# =====================================================================
STARTUPS_URL = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"


@st.cache_data(show_spinner="讀取資料中…")
def read_csv_url(url):
    return pd.read_csv(url)


@st.cache_data
def load_startups():
    try:
        return read_csv_url(STARTUPS_URL), STARTUPS_URL
    except Exception:  # offline fallback: copy kept in the repo
        return pd.read_csv(os.path.join(BASE, "data", "50_Startups.csv")), "data/50_Startups.csv（本機備份）"


@st.cache_resource(show_spinner="建立模型中…")
def build_pipeline(df_key, _df, target, features, test_size):
    return Pipeline(_df, target, list(features), test_size)


def fmt(v):
    """Readable number: thousands separators for large values, 4 significant digits otherwise."""
    if abs(v) >= 1e9:
        return f"{v / 1e9:,.2f}B"
    if abs(v) >= 1e6:
        return f"{v / 1e6:,.1f}M"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"


def page_generic(df, title, source, default_target=None):
    st.title(title)
    st.caption(f"資料來源：{source} ｜ 自動依 CRISP-DM 完成資料理解、準備、特徵選擇（只用訓練資料）、"
               "多元線性回歸、評估與預測")

    # ---- Data Understanding ----
    st.subheader("1. 資料理解")
    a, b, c = st.columns(3)
    a.metric("筆數", f"{len(df):,}")
    b.metric("欄位數", f"{df.shape[1]}")
    c.metric("缺失值", f"{int(df.isna().sum().sum()):,}")
    with st.expander("資料預覽與欄位型態", expanded=False):
        st.dataframe(df.head(20), width="stretch")
        st.dataframe(pd.DataFrame({"型態": df.dtypes.astype(str), "缺失值": df.isna().sum(),
                                   "不同值個數": df.nunique()}), width="stretch")

    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        st.error("找不到數值欄位，無法做回歸。")
        return
    with st.sidebar:
        st.subheader("模型設定")
        t_default = default_target if default_target in numeric else numeric[-1]
        target = st.selectbox("目標欄位（要預測的值）", numeric, index=numeric.index(t_default))
        guess = guess_feature_columns(df, target)
        features = st.multiselect("使用的特徵欄位", [c for c in df.columns if c != target], default=guess)
        test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.05)
    if not features:
        st.warning("請至少選一個特徵欄位。")
        return

    df_key = f"{pd.util.hash_pandas_object(df, index=True).sum()}"
    try:
        pipe = build_pipeline(df_key, df, target, tuple(features), test_size)
    except Exception as e:
        st.error(f"資料準備失敗：{e}")
        return
    n_design = pipe.Xtr.shape[1]
    with st.sidebar:
        method = st.selectbox("特徵選擇方法", METHODS, index=2)
        if method == "Sequential FS" and n_design * len(pipe.ytr) > 2_000_000:
            st.caption("⚠️ 特徵或資料很多時 Sequential FS 會比較慢（需逐步做交叉驗證）。")
        k = st.slider("保留特徵數 k", 1, n_design, min(n_design, max(1, n_design // 2 + 1)),
                      disabled=method.startswith("All-in"))

    # ---- Data Preparation ----
    st.subheader("2. 資料準備")
    st.markdown(f"- 刪除目標欄位缺失的資料 **{pipe.n_rows_dropped}** 筆；數值欄位缺失值用訓練集中位數補值\n"
                f"- 類別欄位 {', '.join(pipe.cat_cols) or '（無）'} 做 one-hot 編碼並去掉一個基準類別（避開 dummy variable trap）\n"
                f"- 訓練集 **{len(pipe.ytr)}** 筆 / 測試集 **{len(pipe.yte)}** 筆；編碼後共 **{n_design}** 個候選特徵")

    # ---- Modeling ----
    try:
        cols = pipe.select(method, k)
        pipe.fit(cols)
        ev, sf = pipe.evaluate()
    except Exception as e:
        st.error(f"建模失敗：{e}")
        return
    st.subheader("3. 特徵選擇與建模")
    st.markdown(f"**{method}** 從訓練資料選出 **{len(cols)}** 個特徵：`{'`, `'.join(cols)}`")
    coef = pd.DataFrame({"係數": pipe.model.params, "p 值": pipe.model.pvalues,
                         "95% CI 下限": pipe.model.conf_int()[0], "95% CI 上限": pipe.model.conf_int()[1]})
    st.dataframe(coef.style.format("{:.4g}"), width="stretch")

    # ---- Evaluation ----
    st.subheader("4. 評估（測試集）")
    m = st.columns(6)
    m[0].metric("MAE", fmt(ev["MAE"]))
    m[1].metric("MSE", fmt(ev["MSE"]))
    m[2].metric("RMSE", fmt(ev["RMSE"]))
    m[3].metric("R²", f"{ev['R2']:.3f}")
    m[4].metric("Adjusted R²", f"{ev['AdjR2']:.3f}")
    m[5].metric("95% PI 覆蓋率", f"{ev['PI95_coverage']:.0f}%")
    gap = ev["train_R2"] - ev["R2"]
    st.caption(f"訓練集 R² {ev['train_R2']:.3f} vs 測試集 R² {ev['R2']:.3f}（差 {gap:+.3f}）："
               + ("差距很小，沒有 overfit。" if gap < 0.05 else
                  "差距不大，屬正常範圍（測試集筆數少時 R² 波動較大）。" if gap < 0.1 else
                  "訓練明顯優於測試，可能 overfit，可以減少 k。"))

    plot = pd.DataFrame({"實際值": pipe.yte.values, "預測值": sf["mean"].values,
                         "PI 下限": sf["obs_ci_lower"].values, "PI 上限": sf["obs_ci_upper"].values})
    plot = plot.sort_values("預測值").reset_index(drop=True)
    plot["測試樣本（依預測值排序）"] = plot.index + 1
    base = alt.Chart(plot).encode(x=alt.X("測試樣本（依預測值排序）:Q"))
    band = base.mark_area(opacity=0.25, color="#1baf7a").encode(y=alt.Y("PI 下限:Q", title=target), y2="PI 上限:Q")
    line = base.mark_line(color="#2a78d6", strokeWidth=2).encode(y="預測值:Q")
    pts = base.mark_circle(color="#0b0b0b", size=28).encode(y="實際值:Q", tooltip=["實際值", "預測值", "PI 下限", "PI 上限"])
    st.markdown("**預測圖**：藍線＝預測值、綠色帶＝95% 預測區間、黑點＝實際值")
    st.altair_chart((band + line + pts).properties(height=340), width="stretch")

    with st.expander("比較所有特徵選擇方法（同一個 k）"):
        if st.button("開始比較"):
            st.dataframe(pipe.compare_methods(k).style.format({"Test R²": "{:.3f}", "Test RMSE": "{:,.4g}"}, na_rep="–"),
                         width="stretch")

    # ---- Deployment ----
    st.subheader("5. 部署：預測新樣本")
    used = [f for f in features if f in pipe.cat_cols and any(c.startswith(f + "_") for c in cols)
            or f in pipe.num_cols and f in cols]
    with st.form("new_sample"):
        raw = {}
        fcols = st.columns(min(4, max(1, len(used))))
        for i, f in enumerate(used):
            with fcols[i % len(fcols)]:
                if f in pipe.cat_cols:
                    raw[f] = st.selectbox(f, pipe.levels[f])
                else:
                    raw[f] = st.number_input(f, value=float(pipe.medians[f]))
        st.form_submit_button("預測")
    for f in features:  # features not used by the model keep their training median / first level
        raw.setdefault(f, pipe.levels[f][0] if f in pipe.cat_cols else float(pipe.medians[f]))
    r = pipe.predict_row(raw)
    st.success(f"預測 {target} = **{fmt(r['mean'])}**　｜　95% 預測區間 {fmt(r['obs_ci_lower'])} ～ {fmt(r['obs_ci_upper'])}")


# =====================================================================
# Problem selector (dropdown)
# =====================================================================
PROBLEMS = ["① King County 房價（主模型）", "② 50 Startups 利潤", "③ 自行輸入資料集"]
with st.sidebar:
    st.header("📈 HW2 多元線性回歸")
    try:  # ?problem=1|2|3 in the URL preselects a problem (handy for sharing a direct link)
        start = min(max(int(st.query_params.get("problem", 1)) - 1, 0), len(PROBLEMS) - 1)
    except ValueError:
        start = 0
    problem = st.selectbox("選擇問題 Problem", PROBLEMS, index=start)
    st.divider()

if problem == PROBLEMS[0]:
    page_king_county()
elif problem == PROBLEMS[1]:
    df_s, src = load_startups()
    page_generic(df_s, "💼 50 Startups 利潤預測", src, default_target="Profit")
else:
    st.title("📂 自行輸入資料集")
    how = st.radio("資料來源", ["CSV 網址", "上傳 CSV 檔"], horizontal=True)
    df_u, src = None, None
    if how == "CSV 網址":
        url = st.text_input("CSV 網址（例如 GitHub raw 連結）", value=STARTUPS_URL)
        if url:
            try:
                df_u, src = read_csv_url(url.strip()), url.strip()
            except Exception as e:
                st.error(f"讀取失敗：{e}")
    else:
        up = st.file_uploader("上傳 CSV", type=["csv"])
        if up is not None:
            df_u, src = pd.read_csv(up), up.name
    if df_u is not None:
        page_generic(df_u, "📊 自訂資料集分析", src)
    else:
        st.info("輸入 CSV 網址或上傳檔案後，網站會自動完成資料理解、特徵選擇、建模、評估與預測。")
