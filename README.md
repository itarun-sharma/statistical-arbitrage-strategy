# Statistical Arbitrage & Pairs Trading Research Platform

An end-to-end quantitative research project for **discovering, validating, and backtesting statistical arbitrage opportunities in Indian equities** using pairs trading, OLS regression, Augmented Dickey-Fuller (ADF) testing, Johansen cointegration, Z-score based trading signals, and walk-forward validation.

> **Research project — historical backtest simulations only. Not investment advice.**

---

## Live Demo

**[Live Streamlit Dashboard](https://statistical-arbitrage-strategy.streamlit.app/)**


## 📌 Overview

This project searches a universe of **65 Indian stocks** for statistically related pairs that may exhibit mean-reverting behavior.

The research pipeline follows:

```text
Historical Market Data
        ↓
   OLS Regression
        ↓
    ADF Testing
        ↓
 Candidate Pair Selection
        ↓
 Johansen Cointegration
        ↓
   Spread Analysis
        ↓
 Z-Score Trading Signals
        ↓
 Out-of-Sample Backtest
        ↓
 Sensitivity Analysis
        ↓
 Walk-Forward Validation
```

The objective is to determine whether a pair of stocks has a sufficiently stable statistical relationship that can potentially be exploited through **market-neutral pairs trading**.

---

## 🔬 Research Configuration

| Parameter                   |                   Value |
| --------------------------- | ----------------------: |
| Stock Universe              |               65 stocks |
| Possible Pairs              |                   2,080 |
| Historical Data             | 2015-01-01 → 2026-01-01 |
| Training Period End         |              2022-12-31 |
| ADF p-value Cutoff          |                    0.10 |
| Maximum Johansen Candidates |                      20 |
| Entry Z-Score               |                     1.5 |
| Exit Z-Score                |                     0.0 |
| Rolling Window              |                 20 days |
| Transaction Cost            |                   5 bps |

These parameters are taken directly from the research run.

---

# 🧠 Methodology

## 1. Stock Universe

The system analyzes **65 Indian equities**, generating up to:

[
\frac{65\times64}{2}=2080
]

unique stock pairs.

Historical data is divided into training and testing periods.

The run produced **1,976 training observations** and **740 testing observations**.

---

## 2. OLS Regression

For each pair, Ordinary Least Squares (OLS) regression is used to estimate the relationship between the two stocks.

A simplified model is:

[
Y_t = \alpha + \beta X_t + \epsilon_t
]

where:

* (Y_t) = price of Stock 1
* (X_t) = price of Stock 2
* (\alpha) = intercept
* (\beta) = hedge ratio
* (\epsilon_t) = residual/spread

The residual becomes the basis for testing whether the relationship is mean reverting.

---

## 3. Augmented Dickey-Fuller (ADF) Test

The ADF test is applied to the residual spread.

The objective is to determine whether the spread is **stationary**.

Conceptually:

```text
Non-stationary spread
       ↓
   ADF test
       ↓
Stationary spread?
       ↓
Potential mean-reverting pair
```

A lower ADF p-value provides stronger evidence against the null hypothesis of a unit root.

The research used an ADF cutoff of **0.10**.

---

## 4. Pair Ranking

Pairs are ranked according to their ADF p-values.

The strongest candidate in the research run was:

### HDFCBANK.NS / KOTAKBANK.NS

| Metric        |    Value |
| ------------- | -------: |
| ADF Statistic |  -4.9224 |
| ADF p-value   | 0.000032 |
| Beta          |   0.5170 |
| Alpha         |   7.0044 |

---

# 5. Johansen Cointegration Validation

ADF identifies potentially stationary relationships, but the project additionally validates candidate pairs using the **Johansen cointegration test**.

The research selected the top 20 candidates for Johansen testing.

Pairs with **Johansen rank = 1** were considered validated cointegrated pairs.

The research found:

```text
394 ADF candidates
        ↓
20 Johansen candidates
        ↓
8 validated rank-1 pairs
```

---

# ⭐ Selected Pair

The selected pair was:

## HDFCBANK.NS / KOTAKBANK.NS

### Cointegration

```text
ADF p-value       : 0.000032
Johansen rank     : 1
```

Johansen cointegration weights:

```text
HDFCBANK.NS : -0.953823
KOTAKBANK.NS:  1.000000
```

The resulting spread showed:

| Diagnostic           |     Result |
| -------------------- | ---------: |
| Training ADF p-value |   0.000096 |
| OOS ADF p-value      |   0.013998 |
| Training Half-Life   | 25.12 days |

---

# 📈 Trading Strategy

The strategy uses the spread's rolling Z-score to generate trading signals.

The basic idea is:

```text
Spread moves far from mean
          ↓
      Z-score rises
          ↓
      Enter trade
          ↓
Spread mean reverts
          ↓
      Z-score → 0
          ↓
      Exit trade
```

### Parameters

```text
Entry Z-score : 1.5
Exit Z-score  : 0.0
Rolling Window: 20 days
Transaction Cost: 5 bps
```

The strategy attempts to exploit **relative mispricing between the two stocks**, rather than simply predicting whether the overall market will rise or fall.

---

# 🧪 Out-of-Sample Backtest

The selected pair was evaluated on previously unseen data.

### Test Period

```text
2023-01-02 → 2025-12-31
```

### Results

| Metric            |     Result |
| ----------------- | ---------: |
| Total Return      | **19.13%** |
| Annualized Return |  **6.14%** |
| Volatility        |  **9.01%** |
| Sharpe Ratio      |   **0.71** |
| Maximum Drawdown  | **-8.69%** |
| Number of Trades  |     **38** |
| Winning Days      |    **241** |
| Losing Days       |    **235** |

---

# 📊 Entry Z-Score Sensitivity

The strategy was tested with different entry thresholds.

| Entry Z | OOS Return | Annual Return |   Sharpe | Max Drawdown | Trades |
| ------: | ---------: | ------------: | -------: | -----------: | -----: |
|     0.5 |     30.40% |         9.46% |     0.91 |      -11.30% |     62 |
|     1.0 |     15.48% |         5.02% |     0.56 |       -9.22% |     48 |
| **1.5** | **19.13%** |     **6.14%** | **0.71** |   **-8.69%** | **38** |
|     2.0 |     19.41% |         6.23% |     0.82 |       -8.69% |     26 |

This demonstrates how changing the entry threshold affects:

* Return
* Trade frequency
* Sharpe ratio
* Drawdown

---

# 🔄 Walk-Forward Validation

To reduce the risk of relying on a single train/test split, the project also performs **walk-forward validation**.

The research contains four folds:

```text
Fold 1
Train ≤ 2018
Test  → 2019–2020

Fold 2
Train ≤ 2020
Test  → 2021–2022

Fold 3
Train ≤ 2022
Test  → 2023–2024

Fold 4
Train ≤ 2024
Test  → 2025
```

Each fold performs a fresh pair scan instead of assuming that the relationship discovered in one period remains optimal forever.

### Example Results

Fold 3 produced:

| Pair                 | OOS Return | Annual Return |   Sharpe |  Max DD |
| -------------------- | ---------: | ------------: | -------: | ------: |
| HDFCBANK / KOTAKBANK |     26.15% |        12.66% | **1.45** |  -4.74% |
| ABB / SBIN           |      8.22% |         4.14% |     0.36 | -13.20% |
| GRASIM / TATASTEEL   |     35.80% |        17.00% | **1.71** | -11.36% |
| ITC / NTPC           |    -14.23% |        -7.58% |    -0.54 | -27.93% |

The walk-forward results also demonstrate an important characteristic of statistical arbitrage:

> A statistically significant relationship does **not** guarantee profitable future trading performance.

Some highly cointegrated pairs produced negative out-of-sample returns.

---

# 🏆 Research Summary

The complete research pipeline produced:

| Stage                |                   Result |
| -------------------- | -----------------------: |
| Stocks Analyzed      |                   **65** |
| Possible Pairs       |                **2,080** |
| ADF Candidates       |                  **394** |
| Johansen Candidates  |                   **20** |
| Validated Pairs      |                    **8** |
| Selected Pair        | **HDFCBANK / KOTAKBANK** |
| OOS Return           |               **19.13%** |
| OOS Sharpe           |                 **0.71** |
| OOS Maximum Drawdown |               **-8.69%** |

---

# 🛠️ Technologies & Concepts

### Programming

* Python
* NumPy
* Pandas
* SciPy
* Statsmodels
* Matplotlib

### Quantitative Finance

* Statistical Arbitrage
* Pairs Trading
* Cointegration
* Mean Reversion
* Hedge Ratios
* Spread Modeling
* Z-Scores
* Backtesting
* Transaction Costs
* Walk-Forward Validation

### Statistical Methods

* Ordinary Least Squares (OLS)
* Augmented Dickey-Fuller (ADF)
* Johansen Cointegration Test
* Rolling Statistics

---

# 📂 Project Pipeline

```text
                    ┌─────────────────────┐
                    │ Historical Prices   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Generate Pairs     │
                    │    65 → 2,080       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    OLS Regression   │
                    │  Estimate Hedge β   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      ADF Test       │
                    │ Stationarity Filter │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Johansen Validation │
                    │ Cointegration Rank  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Spread + Z-Score    │
                    │ Trading Signals     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ OOS Backtesting     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Sensitivity +       │
                    │ Walk-Forward Tests  │
                    └─────────────────────┘
```

---

# 🚀 Key Takeaways

1. **Statistical significance is not the same as profitability.**
2. ADF can identify stationary residual relationships, but additional validation is useful.
3. Johansen testing provides another perspective on cointegration.
4. Out-of-sample testing is essential for evaluating whether a strategy generalizes.
5. Walk-forward validation helps test whether the strategy remains useful across different market regimes.
6. Transaction costs must be included when evaluating a trading strategy.
7. Different Z-score thresholds produce significantly different risk/return profiles.

---

# ⚠️ Disclaimer

This repository is an **academic/quantitative research project**.

All reported performance figures are based on historical backtest simulations and **do not represent guaranteed future returns**.

This project is **not investment advice** and should not be used as the sole basis for making financial decisions.

---

## 📌 Current Research Result

The strongest pair identified in the primary research run was:

```text
HDFCBANK.NS
      +
KOTAKBANK.NS

ADF p-value       : 0.000032
Johansen rank     : 1
OOS Return        : 19.13%
Annualized Return : 6.14%
Sharpe Ratio      : 0.71
Max Drawdown      : -8.69%
```

**Research complete.**
