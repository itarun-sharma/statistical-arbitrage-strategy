import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import itertools
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

st.set_page_config(
    page_title="Statistical Arbitrage Dashboard",
    page_icon="📈",
    layout="wide",
)

DEFAULT_TICKERS = [
    # Banking & Financials
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS",
    "CANBK.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "BAJFINANCE.NS",
    "BAJAJFINSV.NS", "SBILIFE.NS", "HDFCLIFE.NS",

    # IT
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
     "MPHASIS.NS", "PERSISTENT.NS",

    # Energy, Oil & Gas
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
    "COALINDIA.NS", "BPCL.NS", "IOC.NS", "GAIL.NS",

    # Consumer & FMCG
    "ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS",
    "TATACONSUM.NS", "ASIANPAINT.NS", "TITAN.NS", "TRENT.NS",

    # Automobiles
    "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS",
    "EICHERMOT.NS", "HEROMOTOCO.NS",

    # Industrials & Infrastructure
    "LT.NS", "ADANIENT.NS", "ADANIPORTS.NS", "SIEMENS.NS",
    "BEL.NS", "HAL.NS", "ABB.NS", "BHEL.NS",

    # Pharma & Healthcare
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    "APOLLOHOSP.NS", "LUPIN.NS",

    # Telecom & Other Large Caps
    "BHARTIARTL.NS", "JIOFIN.NS", "ULTRACEMCO.NS", "GRASIM.NS",
    "JSWSTEEL.NS", "TATASTEEL.NS", "HINDALCO.NS", "VEDL.NS",
]


def fmt_pct(x):
    return f"{x:.2%}" if pd.notna(x) else "—"


@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers, start, end):
    data = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        threads=False,
        progress=False,
    )

    if data.empty:
        return pd.DataFrame(), list(tickers), []

    # yfinance can return a MultiIndex or a single-level frame.
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"].copy()
        elif "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            return pd.DataFrame(), list(tickers), []
    else:
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        prices = data[[col]].copy()
        prices.columns = [tickers[0]]

    prices = prices.dropna(axis=1, how="all")
    prices = prices.sort_index()
    available = list(prices.columns)
    missing = [t for t in tickers if t not in available]
    return prices, available, missing


def adf_pair(prices, stock1, stock2):
    pair = prices[[stock1, stock2]].dropna()
    if len(pair) < 100:
        raise ValueError("Not enough observations")

    x = sm.add_constant(pair[stock1])
    model = sm.OLS(pair[stock2], x).fit()
    spread = model.resid
    result = adfuller(spread.dropna(), autolag="AIC")

    return {
        "Stock 1": stock1,
        "Stock 2": stock2,
        "ADF Statistic": result[0],
        "ADF p-value": result[1],
        "Beta": model.params.iloc[1],
        "Alpha": model.params.iloc[0],
    }


def scan_pairs(prices):
    rows = []
    for s1, s2 in itertools.combinations(prices.columns, 2):
        try:
            rows.append(adf_pair(prices, s1, s2))
        except Exception:
            pass
    if not rows:
        return pd.DataFrame(columns=["Stock 1", "Stock 2", "ADF Statistic", "ADF p-value", "Beta", "Alpha"])
    return pd.DataFrame(rows).sort_values("ADF p-value").reset_index(drop=True)


def johansen_pair(prices, stock1, stock2, det_order=0, k_ar_diff=1):
    pair = prices[[stock1, stock2]].dropna()
    log_prices = np.log(pair)
    result = coint_johansen(log_prices, det_order=det_order, k_ar_diff=k_ar_diff)

    # 95% critical value is column 1.
    rank = int(result.lr1[0] > result.cvt[0, 1])
    return result, rank


def validate_pairs(prices, adf_table, max_pairs=20, adf_cutoff=0.10):
    rows = []
    candidates = adf_table[adf_table["ADF p-value"] <= adf_cutoff].head(max_pairs)

    for _, row in candidates.iterrows():
        s1, s2 = row["Stock 1"], row["Stock 2"]
        try:
            _, rank = johansen_pair(prices, s1, s2)
            rows.append({
                "Stock 1": s1,
                "Stock 2": s2,
                "ADF p-value": row["ADF p-value"],
                "Johansen rank": rank,
            })
        except Exception:
            rows.append({
                "Stock 1": s1,
                "Stock 2": s2,
                "ADF p-value": row["ADF p-value"],
                "Johansen rank": 0,
            })

    return pd.DataFrame(rows)


def get_weights(prices, stock1, stock2):
    pair = prices[[stock1, stock2]].dropna()
    log_prices = np.log(pair)
    result = coint_johansen(log_prices, det_order=0, k_ar_diff=1)
    vector = np.real(result.evec[:, 0])

    # Normalize second asset to +1.
    if abs(vector[1]) < 1e-12:
        raise ValueError("Unable to normalize cointegration vector")
    vector = vector / vector[1]
    return vector


def half_life(spread):
    s = spread.dropna()
    if len(s) < 30:
        return np.nan
    lag = s.shift(1).dropna()
    delta = s.diff().dropna()
    lag = lag.loc[delta.index]
    model = sm.OLS(delta, sm.add_constant(lag)).fit()
    beta = model.params.iloc[1]
    if beta >= 0:
        return np.inf
    return float(-np.log(2) / beta)


def make_spread(prices, weights):
    return np.log(prices).dot(weights)


def backtest(prices, weights, entry_z=1.5, exit_z=0.0, window=20, cost_bps=5):
    pair = prices.iloc[:, :2].dropna()
    spread = make_spread(pair, weights)
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    z = (spread - mean) / std.replace(0, np.nan)

    signal = pd.Series(0.0, index=spread.index)
    state = 0.0
    for i in range(len(z)):
        zi = z.iloc[i]
        if not np.isfinite(zi):
            signal.iloc[i] = state
            continue
        if state == 0:
            if zi < -entry_z:
                state = 1.0
            elif zi > entry_z:
                state = -1.0
        elif state == 1 and zi >= exit_z:
            state = 0.0
        elif state == -1 and zi <= exit_z:
            state = 0.0
        signal.iloc[i] = state

    # Use normalized dollar-neutral weights for portfolio returns.
    portfolio_weights = np.asarray(weights, dtype=float)
    portfolio_weights = portfolio_weights / np.sum(np.abs(portfolio_weights))
    log_ret = np.log(pair).diff().fillna(0)
    raw_ret = log_ret.mul(portfolio_weights, axis=1).sum(axis=1)
    position = signal.shift(1).fillna(0)
    turnover = position.diff().abs().fillna(position.abs())
    strategy_ret = position * raw_ret - turnover * (cost_bps / 10000.0)

    equity = (1 + strategy_ret).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1

    n = max(len(strategy_ret), 1)
    years = n / 252
    total_return = equity.iloc[-1] - 1
    annual_return = equity.iloc[-1] ** (1 / years) - 1 if equity.iloc[-1] > 0 and years > 0 else np.nan
    volatility = strategy_ret.std() * np.sqrt(252)
    sharpe = (strategy_ret.mean() / strategy_ret.std()) * np.sqrt(252) if strategy_ret.std() > 0 else np.nan
    trades = int(((signal != 0) & (signal.shift(1).fillna(0) == 0)).sum())

    metrics = {
        "Total Return": total_return,
        "Annualized Return": annual_return,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": drawdown.min(),
        "Number of Trades": trades,
        "Winning Days": int((strategy_ret > 0).sum()),
        "Losing Days": int((strategy_ret < 0).sum()),
    }
    return signal, strategy_ret, equity, z, metrics, spread


def run_walk_forward(full_prices, fold_years=2, min_train_years=4, adf_cutoff=0.10):
    start_year = full_prices.index.min().year
    end_year = full_prices.index.max().year
    rows = []

    # Expanding training window with fixed 2-year test windows.
    test_start_year = start_year + min_train_years
    fold = 1
    while test_start_year <= end_year:
        test_end_year = min(test_start_year + fold_years - 1, end_year)
        train_end = pd.Timestamp(f"{test_start_year - 1}-12-31")
        test_start = pd.Timestamp(f"{test_start_year}-01-01")
        test_end = pd.Timestamp(f"{test_end_year}-12-31")

        train = full_prices.loc[:train_end].dropna(axis=1, how="all")
        test = full_prices.loc[test_start:test_end]
        if len(train) < 300 or len(test) < 30:
            break

        adf = scan_pairs(train)
        candidates = adf[adf["ADF p-value"] <= adf_cutoff].head(10)
        for _, candidate in candidates.iterrows():
            s1, s2 = candidate["Stock 1"], candidate["Stock 2"]
            if s1 not in test.columns or s2 not in test.columns:
                continue
            try:
                _, rank = johansen_pair(train, s1, s2)
                if rank != 1:
                    continue
                weights = get_weights(train, s1, s2)
                pair_test = test[[s1, s2]].dropna()
                if len(pair_test) < 30:
                    continue
                spread_train = make_spread(train[[s1, s2]].dropna(), weights)
                hl = half_life(spread_train)
                _, _, _, _, metrics, _ = backtest(pair_test, weights, entry_z=1.5, exit_z=0, window=20, cost_bps=5)
                rows.append({
                    "Fold": fold,
                    "Stock 1": s1,
                    "Stock 2": s2,
                    "ADF p-value": candidate["ADF p-value"],
                    "Half-life": hl,
                    "OOS Return": metrics["Total Return"],
                    "OOS Annual Return": metrics["Annualized Return"],
                    "OOS Volatility": metrics["Annualized Volatility"],
                    "OOS Sharpe": metrics["Sharpe Ratio"],
                    "OOS Max DD": metrics["Maximum Drawdown"],
                    "Trades": metrics["Number of Trades"],
                })
            except Exception:
                continue
        fold += 1
        test_start_year += fold_years

    return pd.DataFrame(rows)


# ---------------------------- UI ----------------------------
st.title("📈 Statistical Arbitrage / Pairs Trading Dashboard")
st.caption("ADF + Johansen cointegration • strict train/test split • OOS backtest • walk-forward analysis")

with st.sidebar:
    st.header("Configuration")
    tickers = st.multiselect("Universe", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
    data_start = st.date_input("Data start", pd.Timestamp("2015-01-01"))
    data_end = st.date_input("Data end", pd.Timestamp("2026-01-01"))
    train_end = st.date_input("Training end", pd.Timestamp("2022-12-31"))
    adf_cutoff = st.slider("ADF p-value cutoff", 0.01, 0.25, 0.10, 0.01)
    entry_z = st.slider("Entry Z", 0.5, 3.0, 1.5, 0.1)
    exit_z = st.slider("Exit Z", -0.5, 0.5, 0.0, 0.1)
    window = st.slider("Rolling Z-score window", 5, 120, 20, 5)
    cost_bps = st.slider("Transaction cost (bps)", 0, 50, 5, 1)
    run_walk = st.checkbox("Run walk-forward analysis", value=True)
    run_button = st.button("🔄 Run analysis", type="primary", use_container_width=True)

if len(tickers) < 2:
    st.warning("Select at least two stocks.")
    st.stop()

if data_end <= data_start:
    st.error("Data end must be after data start.")
    st.stop()

if not run_button and "prices" not in st.session_state:
    st.info("Configure the universe and parameters in the sidebar, then click **Run analysis**.")
    st.stop()

if run_button:
    with st.spinner("Downloading market data and testing all stock pairs..."):
        prices, available, missing = download_prices(tickers, str(data_start), str(data_end))
    if prices.empty:
        st.error("No price data was downloaded.")
        st.stop()
    st.session_state.prices = prices
    st.session_state.available = available
    st.session_state.missing = missing
    st.session_state.run_config = {
        "train_end": str(train_end),
        "adf_cutoff": adf_cutoff,
        "entry_z": entry_z,
        "exit_z": exit_z,
        "window": window,
        "cost_bps": cost_bps,
        "run_walk": run_walk,
    }

prices = st.session_state.prices
cfg = st.session_state.run_config

# Reuse current session settings so sliders can be changed after download.
train_end_ts = pd.Timestamp(cfg["train_end"])
train = prices.loc[:train_end_ts].dropna(axis=1, how="all")
test = prices.loc[train_end_ts + pd.Timedelta(days=1):].dropna(axis=1, how="all")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Stocks", len(prices.columns))
c2.metric("Possible pairs", len(prices.columns) * (len(prices.columns) - 1) // 2)
c3.metric("Training rows", len(train))
c4.metric("Testing rows", len(test))

if st.session_state.missing:
    st.warning(f"Missing data: {', '.join(st.session_state.missing)}")

st.subheader("1. Universe")
st.dataframe(prices.tail(10), use_container_width=True)

st.subheader("2. ADF Pair Scan")
adf_table = scan_pairs(train)
st.dataframe(
    adf_table.style.format({"ADF Statistic": "{:.4f}", "ADF p-value": "{:.4f}", "Beta": "{:.4f}"}),
    use_container_width=True,
    hide_index=True,
)

st.caption(f"All {len(prices.columns) * (len(prices.columns) - 1) // 2} unique pairs are tested when all selected stocks have data.")

st.subheader("3. Johansen Validation")
validated = validate_pairs(train, adf_table, max_pairs=min(20, len(adf_table)), adf_cutoff=cfg["adf_cutoff"])
if validated.empty:
    st.warning("No ADF candidates reached the configured cutoff.")
    st.stop()
st.dataframe(validated, use_container_width=True, hide_index=True)

actual = validated[validated["Johansen rank"] == 1].copy()
if actual.empty:
    st.warning("No pair passed both the ADF screening and Johansen rank-1 validation in this training period.")
    st.stop()

st.success(f"{len(actual)} validated pair(s) survived. The app can analyze any of them.")

pair_labels = [f"{r['Stock 1']} / {r['Stock 2']}" for _, r in actual.iterrows()]
selected_label = st.selectbox("Select validated pair", pair_labels)
selected_row = actual.iloc[pair_labels.index(selected_label)]
s1, s2 = selected_row["Stock 1"], selected_row["Stock 2"]

weights = get_weights(train, s1, s2)
train_pair = train[[s1, s2]].dropna()
test_pair = test[[s1, s2]].dropna()
train_spread = make_spread(train_pair, weights)
test_spread = make_spread(test_pair, weights)

adf_train = adfuller(train_spread.dropna())
adf_test = adfuller(test_spread.dropna()) if len(test_spread) > 30 else (np.nan, np.nan)
hl = half_life(train_spread)

st.subheader("4. Selected Pair")
a, b, c, d = st.columns(4)
a.metric("Pair", f"{s1} / {s2}")
b.metric("Weight 1", f"{weights[0]:.4f}")
c.metric("Weight 2", f"{weights[1]:.4f}")
d.metric("Training half-life", f"{hl:.1f} days" if np.isfinite(hl) else "∞")

x1, x2 = st.columns(2)
x1.metric("Training ADF p-value", f"{adf_train[1]:.4f}")
x2.metric("OOS ADF p-value", f"{adf_test[1]:.4f}" if np.isfinite(adf_test[1]) else "—")

st.markdown("**Interpretation:** the training spread was selected using only the training period. The OOS ADF test is reported separately and is not used to select the pair.")

st.subheader("5. Out-of-Sample Backtest")
if test_pair.empty:
    st.error("No testing data for the selected pair.")
    st.stop()

signal, strategy_ret, equity, z, metrics, spread = backtest(
    test_pair,
    weights,
    entry_z=cfg["entry_z"],
    exit_z=cfg["exit_z"],
    window=cfg["window"],
    cost_bps=cfg["cost_bps"],
)

m = st.columns(6)
m[0].metric("Total Return", fmt_pct(metrics["Total Return"]))
m[1].metric("Annual Return", fmt_pct(metrics["Annualized Return"]))
m[2].metric("Volatility", fmt_pct(metrics["Annualized Volatility"]))
m[3].metric("Sharpe", f"{metrics['Sharpe Ratio']:.2f}")
m[4].metric("Max Drawdown", fmt_pct(metrics["Maximum Drawdown"]))
m[5].metric("Trades", metrics["Number of Trades"])

st.line_chart(pd.DataFrame({"Equity": equity}))

left, right = st.columns(2)
with left:
    st.markdown("**OOS Spread**")
    st.line_chart(pd.DataFrame({"Spread": spread}))
with right:
    st.markdown("**OOS Z-score**")
    st.line_chart(pd.DataFrame({"Z-score": z, "Entry +": pd.Series(cfg["entry_z"], index=z.index), "Entry -": pd.Series(-cfg["entry_z"], index=z.index)}))

st.subheader("6. Parameter Sensitivity")
sensitivity_rows = []
for ez in [0.5, 1.0, 1.5, 2.0]:
    _, _, _, _, met, _ = backtest(test_pair, weights, entry_z=ez, exit_z=cfg["exit_z"], window=cfg["window"], cost_bps=cfg["cost_bps"])
    sensitivity_rows.append({
        "Entry Z": ez,
        "OOS Return": met["Total Return"],
        "Annual Return": met["Annualized Return"],
        "Sharpe": met["Sharpe Ratio"],
        "Max Drawdown": met["Maximum Drawdown"],
        "Trades": met["Number of Trades"],
    })
sens = pd.DataFrame(sensitivity_rows)
st.dataframe(
    sens.style.format({"OOS Return": "{:.2%}", "Annual Return": "{:.2%}", "Sharpe": "{:.2f}", "Max Drawdown": "{:.2%}"}),
    use_container_width=True,
    hide_index=True,
)

if cfg["run_walk"]:
    st.subheader("7. Walk-Forward OOS")
    with st.spinner("Running expanding-window walk-forward tests..."):
        wf = run_walk_forward(prices, fold_years=2, min_train_years=4, adf_cutoff=cfg["adf_cutoff"])
    if wf.empty:
        st.info("No walk-forward pair survived the validation filters.")
    else:
        st.dataframe(
            wf.style.format({
                "ADF p-value": "{:.4f}",
                "Half-life": "{:.1f}",
                "OOS Return": "{:.2%}",
                "OOS Annual Return": "{:.2%}",
                "OOS Volatility": "{:.2%}",
                "OOS Sharpe": "{:.2f}",
                "OOS Max DD": "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Each fold selects pairs using its training window and evaluates them only on the subsequent test window.")

st.subheader("8. Project Summary")
st.info(
    "This dashboard compares the selected stock universe pairwise, screens with the ADF test, "
    "validates candidates with Johansen rank, estimates the cointegration vector and half-life, "
    "then evaluates the selected relationship out-of-sample with transaction costs and parameter sensitivity. "
    "It is a research/backtesting tool, not a live trading system."
)
