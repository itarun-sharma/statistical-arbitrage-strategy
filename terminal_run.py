import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import itertools
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# ============================================================
# CONFIGURATION
# ============================================================

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

DATA_START = "2015-01-01"
DATA_END = "2026-01-01"
TRAIN_END = "2022-12-31"

ADF_CUTOFF = 0.10
MAX_JOHANSEN_CANDIDATES = 20

ENTRY_Z = 1.5
EXIT_Z = 0.0
ROLLING_WINDOW = 20
COST_BPS = 5

RUN_WALK_FORWARD = True
WALK_FORWARD_FOLD_YEARS = 2
WALK_FORWARD_MIN_TRAIN_YEARS = 4


# ============================================================
# DATA
# ============================================================

def download_prices(tickers, start, end):
    print("\n[1/7] Downloading historical market data...")

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
    missing = [ticker for ticker in tickers if ticker not in available]

    return prices, available, missing


# ============================================================
# ADF + OLS
# ============================================================

def adf_pair(prices, stock1, stock2):
    pair = prices[[stock1, stock2]].dropna()

    if len(pair) < 100:
        raise ValueError("Not enough observations")

    x = sm.add_constant(pair[stock1])

    model = sm.OLS(pair[stock2], x).fit()

    spread = model.resid

    result = adfuller(
        spread.dropna(),
        autolag="AIC"
    )

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

    total_pairs = len(prices.columns) * (len(prices.columns) - 1) // 2
    print(f"\nTesting {total_pairs:,} unique stock pairs using OLS + ADF...")

    for i, (s1, s2) in enumerate(
        itertools.combinations(prices.columns, 2),
        start=1
    ):
        try:
            rows.append(adf_pair(prices, s1, s2))
        except Exception:
            pass

    if not rows:
        return pd.DataFrame(
            columns=[
                "Stock 1",
                "Stock 2",
                "ADF Statistic",
                "ADF p-value",
                "Beta",
                "Alpha",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values("ADF p-value")
        .reset_index(drop=True)
    )


# ============================================================
# JOHANSEN
# ============================================================

def johansen_pair(
    prices,
    stock1,
    stock2,
    det_order=0,
    k_ar_diff=1
):
    pair = prices[[stock1, stock2]].dropna()

    log_prices = np.log(pair)

    result = coint_johansen(
        log_prices,
        det_order=det_order,
        k_ar_diff=k_ar_diff
    )

    # 95% critical value = column 1
    rank = int(result.lr1[0] > result.cvt[0, 1])

    return result, rank


def validate_pairs(
    prices,
    adf_table,
    max_pairs=20,
    adf_cutoff=0.10
):
    candidates = (
        adf_table[
            adf_table["ADF p-value"] <= adf_cutoff
        ]
        .head(max_pairs)
    )

    rows = []

    for _, row in candidates.iterrows():
        s1 = row["Stock 1"]
        s2 = row["Stock 2"]

        try:
            _, rank = johansen_pair(
                prices,
                s1,
                s2
            )

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


# ============================================================
# COINTEGRATION WEIGHTS
# ============================================================

def get_weights(prices, stock1, stock2):
    pair = prices[[stock1, stock2]].dropna()

    log_prices = np.log(pair)

    result = coint_johansen(
        log_prices,
        det_order=0,
        k_ar_diff=1
    )

    vector = np.real(result.evec[:, 0])

    # Normalize second asset to +1
    if abs(vector[1]) < 1e-12:
        raise ValueError(
            "Unable to normalize cointegration vector"
        )

    vector = vector / vector[1]

    return vector


# ============================================================
# HALF-LIFE + SPREAD
# ============================================================

def half_life(spread):
    s = spread.dropna()

    if len(s) < 30:
        return np.nan

    lag = s.shift(1).dropna()
    delta = s.diff().dropna()

    lag = lag.loc[delta.index]

    model = sm.OLS(
        delta,
        sm.add_constant(lag)
    ).fit()

    beta = model.params.iloc[1]

    if beta >= 0:
        return np.inf

    return float(-np.log(2) / beta)


def make_spread(prices, weights):
    return np.log(prices).dot(weights)


# ============================================================
# BACKTEST
# ============================================================

def backtest(
    prices,
    weights,
    entry_z=1.5,
    exit_z=0.0,
    window=20,
    cost_bps=5
):
    pair = prices.iloc[:, :2].dropna()

    spread = make_spread(
        pair,
        weights
    )

    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()

    z = (
        (spread - mean)
        / std.replace(0, np.nan)
    )

    signal = pd.Series(
        0.0,
        index=spread.index
    )

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

    # Normalize weights by absolute exposure
    portfolio_weights = np.asarray(
        weights,
        dtype=float
    )

    portfolio_weights = (
        portfolio_weights
        / np.sum(np.abs(portfolio_weights))
    )

    log_ret = (
        np.log(pair)
        .diff()
        .fillna(0)
    )

    raw_ret = (
        log_ret
        .mul(portfolio_weights, axis=1)
        .sum(axis=1)
    )

    position = (
        signal
        .shift(1)
        .fillna(0)
    )

    turnover = (
        position
        .diff()
        .abs()
        .fillna(position.abs())
    )

    strategy_ret = (
        position * raw_ret
        - turnover * (cost_bps / 10000.0)
    )

    equity = (
        1 + strategy_ret
    ).cumprod()

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
        - 1
    )

    n = max(
        len(strategy_ret),
        1
    )

    years = n / 252

    total_return = (
        equity.iloc[-1] - 1
    )

    annual_return = (
        equity.iloc[-1] ** (1 / years) - 1
        if equity.iloc[-1] > 0 and years > 0
        else np.nan
    )

    volatility = (
        strategy_ret.std()
        * np.sqrt(252)
    )

    sharpe = (
        strategy_ret.mean()
        / strategy_ret.std()
        * np.sqrt(252)
        if strategy_ret.std() > 0
        else np.nan
    )

    trades = int(
        (
            (signal != 0)
            & (signal.shift(1).fillna(0) == 0)
        ).sum()
    )

    metrics = {
        "Total Return": total_return,
        "Annualized Return": annual_return,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": drawdown.min(),
        "Number of Trades": trades,
        "Winning Days": int(
            (strategy_ret > 0).sum()
        ),
        "Losing Days": int(
            (strategy_ret < 0).sum()
        ),
    }

    return (
        signal,
        strategy_ret,
        equity,
        z,
        metrics,
        spread
    )


# ============================================================
# WALK-FORWARD
# ============================================================

def run_walk_forward(
    full_prices,
    fold_years=2,
    min_train_years=4,
    adf_cutoff=0.10
):
    start_year = full_prices.index.min().year
    end_year = full_prices.index.max().year

    rows = []

    test_start_year = (
        start_year
        + min_train_years
    )

    fold = 1

    while test_start_year <= end_year:

        test_end_year = min(
            test_start_year
            + fold_years
            - 1,
            end_year
        )

        train_end = pd.Timestamp(
            f"{test_start_year - 1}-12-31"
        )

        test_start = pd.Timestamp(
            f"{test_start_year}-01-01"
        )

        test_end = pd.Timestamp(
            f"{test_end_year}-12-31"
        )

        train = (
            full_prices
            .loc[:train_end]
            .dropna(axis=1, how="all")
        )

        test = full_prices.loc[
            test_start:test_end
        ]

        if len(train) < 300 or len(test) < 30:
            break

        print(
            f"\nWalk-forward Fold {fold}: "
            f"Train <= {train_end.year} | "
            f"Test {test_start.year}-{test_end.year}"
        )

        adf = scan_pairs(train)

        candidates = (
            adf[
                adf["ADF p-value"] <= adf_cutoff
            ]
            .head(10)
        )

        for _, candidate in candidates.iterrows():

            s1 = candidate["Stock 1"]
            s2 = candidate["Stock 2"]

            if (
                s1 not in test.columns
                or s2 not in test.columns
            ):
                continue

            try:
                _, rank = johansen_pair(
                    train,
                    s1,
                    s2
                )

                if rank != 1:
                    continue

                weights = get_weights(
                    train,
                    s1,
                    s2
                )

                pair_test = (
                    test[[s1, s2]]
                    .dropna()
                )

                if len(pair_test) < 30:
                    continue

                spread_train = make_spread(
                    train[[s1, s2]].dropna(),
                    weights
                )

                hl = half_life(
                    spread_train
                )

                _, _, _, _, metrics, _ = backtest(
                    pair_test,
                    weights,
                    entry_z=ENTRY_Z,
                    exit_z=EXIT_Z,
                    window=ROLLING_WINDOW,
                    cost_bps=COST_BPS
                )

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


# ============================================================
# TERMINAL OUTPUT HELPERS
# ============================================================

def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_pct(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


# ============================================================
# MAIN RESEARCH PIPELINE
# ============================================================

def main():

    print_header(
        "STATISTICAL ARBITRAGE / PAIRS TRADING RESEARCH"
    )

    print("\nConfiguration")
    print("-" * 90)
    print(f"Universe              : {len(DEFAULT_TICKERS)} stocks")
    print(
        f"Possible pairs        : "
        f"{len(DEFAULT_TICKERS) * (len(DEFAULT_TICKERS)-1) // 2:,}"
    )
    print(f"Data                  : {DATA_START} → {DATA_END}")
    print(f"Training end          : {TRAIN_END}")
    print(f"ADF cutoff            : {ADF_CUTOFF}")
    print(f"Max Johansen pairs    : {MAX_JOHANSEN_CANDIDATES}")
    print(f"Entry Z               : {ENTRY_Z}")
    print(f"Exit Z                : {EXIT_Z}")
    print(f"Rolling window        : {ROLLING_WINDOW}")
    print(f"Transaction cost      : {COST_BPS} bps")

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    prices, available, missing = download_prices(
        DEFAULT_TICKERS,
        DATA_START,
        DATA_END
    )

    if prices.empty:
        print("\nERROR: No market data was downloaded.")
        return

    print(f"Stocks downloaded     : {len(available)}")

    if missing:
        print(
            "Missing stocks        : "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # TRAIN / TEST
    # --------------------------------------------------------

    train_end = pd.Timestamp(
        TRAIN_END
    )

    train = (
        prices
        .loc[:train_end]
        .dropna(axis=1, how="all")
    )

    test = (
        prices
        .loc[
            train_end
            + pd.Timedelta(days=1):
        ]
        .dropna(axis=1, how="all")
    )

    print(f"Training observations : {len(train):,}")
    print(f"Testing observations  : {len(test):,}")

    # --------------------------------------------------------
    # ADF
    # --------------------------------------------------------

    print_header(
        "2. ADF + OLS PAIR SCAN"
    )

    adf_table = scan_pairs(train)

    if adf_table.empty:
        print("\nNo valid pairs were found.")
        return

    print("\nTop 20 pairs ranked by ADF p-value:")
    print()

    print(
        adf_table.head(20).to_string(
            index=False,
            formatters={
                "ADF Statistic": lambda x: f"{x:.4f}",
                "ADF p-value": lambda x: f"{x:.6f}",
                "Beta": lambda x: f"{x:.4f}",
                "Alpha": lambda x: f"{x:.4f}",
            }
        )
    )

    # --------------------------------------------------------
    # JOHANSEN
    # --------------------------------------------------------

    print_header(
        "3. JOHANSEN COINTEGRATION VALIDATION"
    )

    validated = validate_pairs(
        train,
        adf_table,
        max_pairs=MAX_JOHANSEN_CANDIDATES,
        adf_cutoff=ADF_CUTOFF
    )

    if validated.empty:
        print(
            "\nNo pair passed the ADF cutoff."
        )
        return

    print(
        validated.to_string(
            index=False,
            formatters={
                "ADF p-value": lambda x: f"{x:.6f}"
            }
        )
    )

    actual = validated[
        validated["Johansen rank"] == 1
    ].copy()

    print(
        f"\nValidated Rank-1 pairs: "
        f"{len(actual)}"
    )

    if actual.empty:
        print(
            "\nNo pair passed both ADF and "
            "Johansen validation."
        )
        return

    # --------------------------------------------------------
    # SELECT BEST VALIDATED PAIR
    # --------------------------------------------------------

    selected = (
        actual
        .sort_values("ADF p-value")
        .iloc[0]
    )

    s1 = selected["Stock 1"]
    s2 = selected["Stock 2"]

    print_header(
        f"4. SELECTED PAIR: {s1} / {s2}"
    )

    print(
        f"ADF p-value         : "
        f"{selected['ADF p-value']:.6f}"
    )

    print(
        f"Johansen rank       : "
        f"{int(selected['Johansen rank'])}"
    )

    # --------------------------------------------------------
    # WEIGHTS
    # --------------------------------------------------------

    weights = get_weights(
        train,
        s1,
        s2
    )

    print("\nJohansen cointegration weights:")
    print(
        f"  {s1:<20}: {weights[0]:.6f}"
    )
    print(
        f"  {s2:<20}: {weights[1]:.6f}"
    )

    # --------------------------------------------------------
    # SPREAD + HALF LIFE
    # --------------------------------------------------------

    train_pair = (
        train[[s1, s2]]
        .dropna()
    )

    test_pair = (
        test[[s1, s2]]
        .dropna()
    )

    train_spread = make_spread(
        train_pair,
        weights
    )

    test_spread = make_spread(
        test_pair,
        weights
    )

    adf_train = adfuller(
        train_spread.dropna()
    )

    if len(test_spread) > 30:
        adf_test = adfuller(
            test_spread.dropna()
        )
    else:
        adf_test = (
            np.nan,
            np.nan
        )

    hl = half_life(
        train_spread
    )

    print("\nSpread diagnostics:")
    print(
        f"Training ADF p-value : "
        f"{adf_train[1]:.6f}"
    )

    if np.isfinite(adf_test[1]):
        print(
            f"OOS ADF p-value      : "
            f"{adf_test[1]:.6f}"
        )
    else:
        print(
            "OOS ADF p-value      : N/A"
        )

    if np.isfinite(hl):
        print(
            f"Training half-life   : "
            f"{hl:.2f} days"
        )
    else:
        print(
            "Training half-life   : ∞"
        )

    # --------------------------------------------------------
    # BACKTEST
    # --------------------------------------------------------

    print_header(
        "5. OUT-OF-SAMPLE BACKTEST"
    )

    if test_pair.empty:
        print(
            "ERROR: No testing data available "
            "for selected pair."
        )
        return

    _, _, equity, _, metrics, _ = backtest(
        test_pair,
        weights,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        window=ROLLING_WINDOW,
        cost_bps=COST_BPS
    )

    print(
        f"Test period          : "
        f"{test_pair.index.min().date()} → "
        f"{test_pair.index.max().date()}"
    )

    print("\nPerformance:")
    print(
        f"  Total Return       : "
        f"{print_pct(metrics['Total Return'])}"
    )
    print(
        f"  Annualized Return  : "
        f"{print_pct(metrics['Annualized Return'])}"
    )
    print(
        f"  Volatility         : "
        f"{print_pct(metrics['Annualized Volatility'])}"
    )
    print(
        f"  Sharpe Ratio       : "
        f"{metrics['Sharpe Ratio']:.2f}"
    )
    print(
        f"  Maximum Drawdown   : "
        f"{print_pct(metrics['Maximum Drawdown'])}"
    )
    print(
        f"  Number of Trades   : "
        f"{metrics['Number of Trades']}"
    )
    print(
        f"  Winning Days       : "
        f"{metrics['Winning Days']}"
    )
    print(
        f"  Losing Days        : "
        f"{metrics['Losing Days']}"
    )

    # --------------------------------------------------------
    # PARAMETER SENSITIVITY
    # --------------------------------------------------------

    print_header(
        "6. ENTRY Z-SCORE SENSITIVITY"
    )

    sensitivity_rows = []

    for ez in [0.5, 1.0, 1.5, 2.0]:

        _, _, _, _, met, _ = backtest(
            test_pair,
            weights,
            entry_z=ez,
            exit_z=EXIT_Z,
            window=ROLLING_WINDOW,
            cost_bps=COST_BPS
        )

        sensitivity_rows.append({
            "Entry Z": ez,
            "OOS Return": met["Total Return"],
            "Annual Return": met["Annualized Return"],
            "Sharpe": met["Sharpe Ratio"],
            "Max Drawdown": met["Maximum Drawdown"],
            "Trades": met["Number of Trades"],
        })

    sens = pd.DataFrame(
        sensitivity_rows
    )

    print(
        sens.to_string(
            index=False,
            formatters={
                "OOS Return": lambda x: f"{x:.2%}",
                "Annual Return": lambda x: f"{x:.2%}",
                "Sharpe": lambda x: f"{x:.2f}",
                "Max Drawdown": lambda x: f"{x:.2%}",
            }
        )
    )

    # --------------------------------------------------------
    # WALK FORWARD
    # --------------------------------------------------------

    if RUN_WALK_FORWARD:

        print_header(
            "7. WALK-FORWARD VALIDATION"
        )

        wf = run_walk_forward(
            prices,
            fold_years=WALK_FORWARD_FOLD_YEARS,
            min_train_years=WALK_FORWARD_MIN_TRAIN_YEARS,
            adf_cutoff=ADF_CUTOFF
        )

        if wf.empty:
            print(
                "\nNo pairs survived walk-forward validation."
            )
        else:
            print(
                "\nWalk-forward results:"
            )

            print(
                wf.to_string(
                    index=False,
                    formatters={
                        "ADF p-value": lambda x: f"{x:.6f}",
                        "Half-life": lambda x: f"{x:.1f}",
                        "OOS Return": lambda x: f"{x:.2%}",
                        "OOS Annual Return": lambda x: f"{x:.2%}",
                        "OOS Volatility": lambda x: f"{x:.2%}",
                        "OOS Sharpe": lambda x: f"{x:.2f}",
                        "OOS Max DD": lambda x: f"{x:.2%}",
                    }
                )
            )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_header(
        "8. RESEARCH SUMMARY"
    )

    print(f"Stocks analyzed       : {len(available)}")
    print(
        "Possible pairs        : "
        f"{len(available) * (len(available)-1) // 2:,}"
    )
    print(
        f"ADF candidates       : "
        f"{len(adf_table[adf_table['ADF p-value'] <= ADF_CUTOFF])}"
    )
    print(
        f"Johansen candidates  : "
        f"{len(validated)}"
    )
    print(
        f"Validated pairs      : "
        f"{len(actual)}"
    )
    print(
        f"Selected pair        : "
        f"{s1} / {s2}"
    )
    print(
        f"OOS return           : "
        f"{print_pct(metrics['Total Return'])}"
    )
    print(
        f"OOS Sharpe           : "
        f"{metrics['Sharpe Ratio']:.2f}"
    )
    print(
        f"OOS max drawdown     : "
        f"{print_pct(metrics['Maximum Drawdown'])}"
    )

    print("\nResearch complete.")
    print(
        "Note: Results are historical backtest simulations, "
        "not investment advice."
    )


if __name__ == "__main__":
    main()