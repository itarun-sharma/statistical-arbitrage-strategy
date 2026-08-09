# Statistical Arbitrage & Pairs Trading Platform

An end-to-end quantitative research project for discovering, validating, and backtesting statistical arbitrage opportunities in Indian equities.

The project combines **ADF residual testing, Johansen cointegration, mean-reversion analysis, walk-forward validation, transaction costs, parameter sensitivity, and an interactive Streamlit dashboard**.

> **Educational / research project:** Backtest results are historical simulations and are not investment advice or a guarantee of future performance.

## Live Demo

**[Live Streamlit Dashboard](https://statistical-arbitrage-strategy.streamlit.app/)**

## Project Overview

The system starts with an 8-stock universe:

- HDFCBANK.NS
- ICICIBANK.NS
- SBIN.NS
- AXISBANK.NS
- RELIANCE.NS
- TCS.NS
- INFY.NS
- ITC.NS

For 8 stocks, the system evaluates all:

**8 choose 2 = 28 unique pairs**

The pipeline then filters candidate pairs using statistical tests before performing out-of-sample backtesting.

## Methodology

```text
Market Data
    ↓
Train / Test Split
    ↓
28 Pair Combinations
    ↓
ADF Residual Test
    ↓
Johansen Cointegration Validation
    ↓
Select Validated Pair
    ↓
Estimate Cointegration Weights
    ↓
Calculate Spread
    ↓
ADF + Half-Life Analysis
    ↓
Trading Signals / Z-Score
    ↓
Transaction Costs
    ↓
Out-of-Sample Backtest
    ↓
Walk-Forward Validation
    ↓
Parameter Sensitivity
    ↓
Streamlit Dashboard
```

## Statistical Tests

### 1. Augmented Dickey-Fuller (ADF)

The ADF test is applied to the residual/spread generated between two assets.

A low p-value provides evidence against the null hypothesis of a unit root in the residual series.

The project uses ADF as an initial pair-screening step.

### 2. Johansen Cointegration Test

Candidate pairs are subsequently evaluated using the Johansen test.

A pair is retained when the Johansen test indicates a cointegrating rank of 1 at the configured significance level.

Using both tests reduces reliance on a single statistical criterion.

## Train / Test Design

The current configuration uses:

| Dataset | Period |
|---|---|
| Training | 2015–2022 |
| Testing | 2023–2025 |

The pair-selection and cointegration parameters are estimated using the training period, while the selected strategy is evaluated on previously unseen testing data.

This separation is intended to reduce look-ahead bias.

## Mean-Reversion Model

The cointegration vector is normalized to construct the trading spread.

The strategy uses a rolling Z-score to identify deviations from the estimated mean:

- **Z-score below entry threshold:** long spread
- **Z-score above entry threshold:** short spread
- **Z-score returns toward the mean:** exit

The dashboard allows the entry threshold to be varied for sensitivity analysis.

## Walk-Forward Validation

The project also evaluates the methodology over multiple expanding training windows and subsequent out-of-sample periods.

Example folds:

```text
Fold 1: Train ≤ 2018 → Test 2019–2020
Fold 2: Train ≤ 2020 → Test 2021–2022
Fold 3: Train ≤ 2022 → Test 2023–2024
Fold 4: Train ≤ 2024 → Test 2025
```

This helps evaluate whether the strategy's behavior is robust across different market periods rather than relying on a single train/test split.

## Transaction Costs

Backtesting includes configurable transaction-cost assumptions so that reported performance is not based solely on frictionless trading.

## Current Example Results

One observed out-of-sample run selected:

**ICICIBANK.NS / TCS.NS**

with approximately:

| Metric | Result |
|---|---:|
| OOS Total Return | 9.43% |
| Annualized Return | 3.12% |
| Annualized Volatility | 11.04% |
| Sharpe Ratio | 0.33 |
| Maximum Drawdown | -17.09% |
| Trades | 58 |

These numbers are **one historical backtest configuration**, not a claim of expected future returns.

The project also evaluates parameter sensitivity. In one run, increasing the entry Z-score changed the observed OOS results, demonstrating why parameter selection should not be based on a single backtest.

## Streamlit Dashboard

The Streamlit interface provides interactive access to the research pipeline.

Typical dashboard functionality includes:

- Stock universe configuration
- Pair screening results
- ADF statistics and p-values
- Johansen validation
- Validated pair selection
- Cointegration weights
- Spread visualization
- Z-score visualization
- Half-life
- OOS performance metrics
- Drawdown analysis
- Trade statistics
- Walk-forward results
- Entry-threshold sensitivity

## Project Structure

```text
statistical-arbitrage-strategy/
│
├── app.py
├── requirements.txt
├── README.md
└── ...
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/statistical-arbitrage-strategy.git
cd statistical-arbitrage-strategy
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit Dashboard

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Run the Research Script

```bash
python cointegration_and_backtesting_enhanced.py
```

The research script downloads historical market data and performs the statistical analysis and backtesting pipeline.

## Technologies

- **Python**
- **Pandas**
- **NumPy**
- **Statsmodels**
- **yfinance**
- **Matplotlib**
- **Streamlit**

## Key Quantitative Concepts

This project demonstrates practical implementation of:

- Statistical arbitrage
- Pairs trading
- Cointegration
- Augmented Dickey-Fuller testing
- Johansen cointegration testing
- OLS hedge-ratio estimation
- Mean reversion
- Z-score signals
- Half-life estimation
- Out-of-sample testing
- Walk-forward validation
- Transaction-cost modeling
- Drawdown analysis
- Sharpe ratio
- Parameter sensitivity analysis

## Limitations

This is a research/backtesting project and has several important limitations:

1. Historical relationships can break down.
2. Cointegration is not guaranteed to remain stable.
3. Daily data does not model intraday execution.
4. Slippage, market impact, liquidity constraints, and borrow availability may differ from assumptions.
5. Statistical significance does not guarantee profitability.
6. Multiple testing across many pairs can create false discoveries.
7. The current universe is small and focused on selected Indian equities.
8. Backtest results should not be interpreted as live-trading performance.

## Future Improvements

Potential extensions include:

- Dynamic hedge-ratio estimation using Kalman filters
- Regime detection
- Johansen rank selection across larger universes
- False-discovery-rate control for multiple pair testing
- More realistic slippage and market-impact models
- Position sizing based on volatility
- Stop-loss / risk controls
- Portfolio-level optimization across multiple pairs
- Intraday data
- Live paper trading
- Broker API integration
- Automated monitoring and alerts

## Disclaimer

This repository is intended for **educational and quantitative research purposes only**. It does not constitute financial, investment, or trading advice. Historical backtest performance does not guarantee future results.
