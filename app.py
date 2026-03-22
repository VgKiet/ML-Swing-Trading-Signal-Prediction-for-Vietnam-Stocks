import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tvDatafeed import TvDatafeed, Interval


# ================= LOAD MODEL =================
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")


# ================= UI =================
st.title("📈 Stock Trading Signal Prediction - ML Logistic Regression")

ticker = st.text_input(
    "Nhập mã cổ phiếu (HPG, VNM, VIC...):",
    "HPG"
).upper().strip()


# ================= LOAD DATA =================
tv = TvDatafeed()

try:
    df = tv.get_hist(
        symbol=ticker,
        exchange="HOSE",
        interval=Interval.in_daily,
        n_bars=5000
    )
except:
    st.error("Không tải được dữ liệu")
    st.stop()

if df is None or df.empty:
    st.warning("Mã không tồn tại trên HOSE")
    st.stop()


df.reset_index(inplace=True)
df.rename(columns={"datetime": "time"}, inplace=True)


for col in ["open", "high", "low", "close"]:
    df[col] = df[col] / 1000


# ================= FEATURE =================
def supertrend_kivanc(df, periods=10, multiplier=7):

    high = df.high
    low = df.low
    close = df.close

    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/periods).mean()

    src = (high + low) / 2

    up = src - multiplier * atr
    dn = src + multiplier * atr

    trend = np.ones(len(df))

    for i in range(1, len(df)):

        if close.iloc[i] > dn.iloc[i-1]:
            trend[i] = 1

        elif close.iloc[i] < up.iloc[i-1]:
            trend[i] = -1

        else:
            trend[i] = trend[i-1]

    df["supertrend_dir"] = trend
    df["supertrend"] = np.where(trend == 1, up, dn)

    return df


def stc(close):

    macd = close.ewm(span=26).mean() - close.ewm(span=50).mean()

    return macd.rolling(12).mean()


def donchian(df):

    high = df.high.rolling(20).max()
    low = df.low.rolling(20).min()

    mid = (high + low) / 2

    df["donchian_trend"] = np.where(
        df.close > mid, 1,
        np.where(df.close < mid, -1, 0)
    )

    return df


df = supertrend_kivanc(df)

df["stc"] = stc(df.close)

df = donchian(df)

df["close_supertrend"] = df.close - df.supertrend


feature_cols = [
    "close",
    "supertrend_dir",
    "close_supertrend",
    "stc",
    "donchian_trend"
]


# ================= PREDICT =================
X = df[feature_cols].dropna()

X_scaled = scaler.transform(X)

y_pred = model.predict(X_scaled)

df.loc[X.index, "ml_signal"] = y_pred


# ================= SIDEWAY FILTER =================
df["range_pct"] = (df.high - df.low) / df.close

df.loc[
    (df.range_pct >= 0.01) &
    (df.range_pct <= 0.015),
    "ml_signal"
] = 0


# ================= DONCHIAN + STC FILTER =================
df_test = df.copy()

df_test["dc_high"] = df_test.high.rolling(20).max()
df_test["dc_low"] = df_test.low.rolling(20).min()

df_test["stc_diff"] = df_test.stc.diff()

df_test["signal_plot"] = 0


for i in range(1, len(df_test)):

    ml = df_test.ml_signal.iloc[i]
    close = df_test.close.iloc[i]
    dc_high = df_test.dc_high.iloc[i]
    dc_low = df_test.dc_low.iloc[i]
    stc_diff = df_test.stc_diff.iloc[i]
    prev = df_test.signal_plot.iloc[i-1]


    # BUY
    if ml == 1 and close <= dc_low * 1.1 and stc_diff > 0 and prev != 1:

        df_test.loc[df_test.index[i], "signal_plot"] = 1


    # SELL
    elif ml == -1 and close >= dc_high * 0.9 and stc_diff < 0 and prev != -1:

        df_test.loc[df_test.index[i], "signal_plot"] = -1


df_test["plot_signal"] = df_test.signal_plot.where(
    df_test.signal_plot != df_test.signal_plot.shift(-1)
)


# ================= FIRST CHART =================
st.subheader("📊 Biểu đồ 1: Signal đã lọc (Donchian + STC)")

plot_data = df_test.tail(1200)

fig1, ax1 = plt.subplots(figsize=(16, 6))

ax1.plot(plot_data.time, plot_data.close, label="Close")

ax1.scatter(
    plot_data.loc[plot_data.plot_signal == 1].time,
    plot_data.loc[plot_data.plot_signal == 1].close,
    marker="^",
    color="green",
    s=140,
    label="BUY (Mũi tên xanh)"
)

ax1.scatter(
    plot_data.loc[plot_data.plot_signal == -1].time,
    plot_data.loc[plot_data.plot_signal == -1].close,
    marker="v",
    color="red",
    s=140,
    label="SELL (Mũi tên đỏ)"
)

ax1.legend()
ax1.grid(alpha=0.3)

ax1.set_title(f"{ticker} ML Trading Signal (Filtered)")

st.pyplot(fig1)


# ================= SECOND CHART =================
st.subheader("📊 Biểu đồ 2: Signal ML")

df_test2 = df.copy()

df_test2["plot_signal"] = df_test2.ml_signal.where(
    df_test2.ml_signal != df_test2.ml_signal.shift(-1)
)

plot_data2 = df_test2.tail(500)

fig2, ax2 = plt.subplots(figsize=(15, 6))

ax2.plot(plot_data2.time, plot_data2.close, label="Close")

ax2.scatter(
    plot_data2.loc[plot_data2.plot_signal == 1].time,
    plot_data2.loc[plot_data2.plot_signal == 1].close,
    marker="^",
    color="green",
    s=120,
    label="BUY (Mũi tên xanh)"
)

ax2.scatter(
    plot_data2.loc[plot_data2.plot_signal == -1].time,
    plot_data2.loc[plot_data2.plot_signal == -1].close,
    marker="v",
    color="red",
    s=120,
    label="SELL (Mũi tên đỏ)"
)

ax2.legend()
ax2.grid(alpha=0.3)

ax2.set_title(f"{ticker} End-of-Sequence ML Signal")

st.pyplot(fig2)

