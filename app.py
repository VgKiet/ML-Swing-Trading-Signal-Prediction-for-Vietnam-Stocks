import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tvDatafeed import TvDatafeed, Interval
import time

# ================= LOAD MODEL =================
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")


# ================= UI =================
tv = TvDatafeed()

st.title('Stock Price Prediction')

user = st.text_input(
    'Nhập mã cổ phiếu (ví dụ: HPG, SHS, BSR...)',
    'HPG'
)

ticker = user.upper().strip()


# Loading message
loading_msg = st.empty()
loading_msg.text("Đang lấy dữ liệu thị trường, vui lòng chờ vài giây...")


# thử lần lượt các sàn
exchanges = ["HOSE", "HNX", "UPCOM"]

df = None
selected_exchange = None


for exchange in exchanges:
    try:
        df = tv.get_hist(
            symbol=ticker,
            exchange=exchange,
            interval=Interval.in_daily,
            n_bars=5000
        )
        if df is not None and not df.empty:
            selected_exchange = exchange
            break
    except:
        continue


loading_msg.empty()


# nếu không tìm thấy mã
if df is None or df.empty:
    st.warning(
        f"Mã cổ phiếu **{ticker}** không tồn tại trên HOSE, HNX hoặc UPCOM.\n"
        "Ví dụ hợp lệ: HPG, FPT, SHS, PVS, BSR, ACV..."
    )
    st.stop()

# Chuẩn hóa dữ liệu
df.reset_index(inplace=True)
df.rename(columns={'datetime': 'time'}, inplace=True)
for col in ['open', 'high', 'low', 'close']:
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

# ================= BACKTEST =================

st.subheader("📊 Backtest lợi nhuận theo tháng")

initial_capital = 100_000_000

BUY_FEE = 0.0015
SELL_FEE = 0.0015
TAX = 0.001

df_bt = df_test.copy()

capital = initial_capital
position = 0
entry_price = 0

trade_log = []
equity_curve = []

for i in range(len(df_bt)):

    signal = df_bt.signal_plot.iloc[i]
    price = df_bt.close.iloc[i]
    time = df_bt.time.iloc[i]

    # ================= BUY =================
    if signal == 1 and position == 0:

        entry_price = price

        shares = capital / (price * (1 + BUY_FEE))

        position = shares

        entry_time = time


    # ================= SELL =================
    elif signal == -1 and position > 0:

        exit_price = price

        capital = position * exit_price * (1 - SELL_FEE - TAX)

        trade_return = (
            exit_price * (1 - SELL_FEE - TAX)
            - entry_price * (1 + BUY_FEE)
        ) / (entry_price * (1 + BUY_FEE))

        trade_log.append({
            "entry_time": entry_time,
            "exit_time": time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "capital": capital,
            "trade_return": trade_return
        })

        position = 0


    # ================= EQUITY UPDATE REALTIME =================
    if position > 0:
        equity_curve.append(position * price)
    else:
        equity_curve.append(capital)


equity_df = pd.DataFrame({
    "time": df_bt.time,
    "equity": equity_curve
})


trades_df = pd.DataFrame(trade_log)


# ================= MONTHLY PROFIT =================

if trades_df.empty:

    st.warning("Chưa có giao dịch để backtest")

else:

    trades_df["month"] = trades_df.exit_time.dt.to_period("M")

    monthly_capital = trades_df.groupby("month")["capital"].last()

    monthly_capital = monthly_capital.reset_index()

    monthly_capital["profit_vnd"] = monthly_capital["capital"].diff()

    monthly_capital.loc[0, "profit_vnd"] = \
        monthly_capital.loc[0, "capital"] - initial_capital


    # ================= RETURN % FIXED =================

    monthly_capital["return_pct"] = (
        monthly_capital["profit_vnd"]
        / monthly_capital["capital"].shift(1)
    ) * 100

    monthly_capital.loc[0, "return_pct"] = (
        monthly_capital.loc[0, "profit_vnd"]
        / initial_capital
    ) * 100


    st.dataframe(monthly_capital)


    # ================= PLOT MONTHLY PROFIT =================

    fig4, ax4 = plt.subplots(figsize=(14,6))

    ax4.bar(
        monthly_capital["month"].astype(str),
        monthly_capital["profit_vnd"]
    )

    ax4.set_title("Monthly Profit (VND)")

    ax4.set_ylabel("Profit (VND)")

    ax4.grid(alpha=0.3)

    plt.xticks(rotation=45)

    st.pyplot(fig4)


    # ================= TOTAL PERFORMANCE =================

    final_capital = equity_curve[-1]

    total_return = (
        (final_capital - initial_capital)
        / initial_capital
    ) * 100


    # ================= MAX DRAWDOWN =================

    equity_series = pd.Series(equity_curve)

    rolling_max = equity_series.cummax()

    drawdown = (
        equity_series - rolling_max
    ) / rolling_max

    max_drawdown = drawdown.min() * 100


    # ================= WINRATE =================

    winrate = (
        trades_df.trade_return > 0
    ).mean() * 100


    # ================= AVG TRADE RETURN =================

    avg_trade_return = trades_df.trade_return.mean() * 100


    # ================= PROFIT FACTOR =================

    gross_profit = trades_df.loc[
        trades_df.trade_return > 0,
        "trade_return"
    ].sum()

    gross_loss = abs(
        trades_df.loc[
            trades_df.trade_return < 0,
            "trade_return"
        ].sum()
    )

    profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0


    # ================= SHARPE RATIO =================

    sharpe_ratio = (
        trades_df.trade_return.mean()
        / trades_df.trade_return.std()
        * np.sqrt(len(trades_df))
        if trades_df.trade_return.std() != 0
        else 0
    )


    # ================= SUMMARY TABLE =================

    st.subheader("📊 Tổng kết hiệu suất Strategy")

    summary_data = {
        "Metric": [
            "Vốn ban đầu",
            "Vốn cuối",
            "Total Return (%)",
            "Max Drawdown (%)",
            "Winrate (%)",
            "Profit Factor",
            "Avg Trade Return (%)",
            "Sharpe Ratio"
        ],
        "Value": [
            f"{initial_capital:,.0f} VND",
            f"{final_capital:,.0f} VND",
            f"{total_return:.2f}",
            f"{max_drawdown:.2f}",
            f"{winrate:.2f}",
            f"{profit_factor:.2f}",
            f"{avg_trade_return:.2f}",
            f"{sharpe_ratio:.2f}"
        ]
    }

    summary_df = pd.DataFrame(summary_data)

    st.dataframe(summary_df, use_container_width=True)

    # ================= EQUITY CURVE =================

    st.subheader("📈 Equity Curve")

    fig5, ax5 = plt.subplots(figsize=(15,6))

    ax5.plot(
        equity_df.time,
        equity_df.equity
    )

    ax5.set_title("Equity Curve")

    ax5.grid(alpha=0.3)

    st.pyplot(fig5)