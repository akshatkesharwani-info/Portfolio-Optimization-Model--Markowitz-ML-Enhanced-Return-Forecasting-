# Portfolio Optimization Model (Markowitz + ML-Enhanced Return Forecasting)

| ML Engineer Track |

100% free and open-source tools only — no paid APIs, no credit card, runs on Google Colab's free tier.

---

## 📌 Overview

Classical Markowitz optimization uses historical average returns as the return forecast, which is often a poor predictor going forward. Build a portfolio optimizer that replaces the naive historical-mean return input with a machine-learning return forecast, then compares the resulting efficient frontier to the classical version -- using only free tools.

**Domain:** Asset Management / Quant Portfolio Construction

---

## 📊 Dataset

Free daily price history for a multi-asset universe via yfinance

**Source:** [https://finance.yahoo.com](https://finance.yahoo.com)

> This script also includes a **safe fallback**: if the real dataset file isn't found next to the
> notebook/script, it automatically generates a small realistic sample dataset with the same column
> names, so the whole pipeline still runs end-to-end even before you've downloaded the real data.

---

## 🛠️ Tech Stack

Python 3 | PyPortfolioOpt (free) | yfinance | scikit-learn | NumPy

**Skills demonstrated:** Python, PyPortfolioOpt (free), scikit-learn, SciPy, pandas

---

## 🎯 What This Project Builds

- Free multi-asset price data loading and return/covariance computation
- A classical Markowitz mean-variance efficient frontier as the baseline
- An ML return forecaster (Random Forest regressor on lagged returns + momentum features) per asset
- An ML-enhanced efficient frontier using the forecasted (instead of historical mean) returns
- A max-Sharpe portfolio comparison: classical vs ML-enhanced weights
- A backtest of both portfolios' realized returns on a held-out period

---

## 🧭 Step-by-Step Approach

### Step 1: Compute Classical Inputs

**What:** Calculate historical mean returns and the covariance matrix for the asset universe

**Why:** This is the standard Markowitz input set and serves as the baseline for comparison

**How:** expected_returns.mean_historical_return(prices); risk_models.sample_cov(prices) from PyPortfolioOpt


### Step 2: Build ML Return Forecaster

**What:** Train a Random Forest regressor per asset on lagged returns and momentum features to predict next-period return

**Why:** Historical mean return is a famously noisy forecast; even a modest ML model can improve on it

**How:** RandomForestRegressor().fit(lagged_features, next_period_return) per ticker


### Step 3: Optimize Both Frontiers

**What:** Run mean-variance optimization once with historical means and once with ML-forecasted means

**Why:** Comparing both frontiers isolates exactly how much the return forecast choice changes the optimal portfolio

**How:** EfficientFrontier(mu, S).max_sharpe() for each mu variant


### Step 4: Backtest Both Portfolios

**What:** Apply both sets of optimal weights to a held-out future period and compare realized returns

**Why:** Backtesting is the only real test of whether the ML-enhanced forecast actually helped out-of-sample

**How:** portfolio_return = (weights * holdout_returns).sum(axis=1)


---

## 📈 Dashboard / Reporting Ideas

- Bar chart: portfolio weights side by side, classical vs ML-enhanced
- Line chart: cumulative holdout return, classical vs ML-enhanced portfolio
- Efficient frontier plot: risk vs return curve for both approaches with the chosen portfolio marked
- KPI cards: expected annual return and volatility for each portfolio version
- Table: per-asset ML-forecasted return vs historical mean return, to see where the forecast diverged most

---

## 💡 Key Insights

- The classical Markowitz result is extremely sensitive to the input mean-return estimate -- this is its best-known weakness
- ML-enhanced forecasts don't need to be highly accurate to shift the optimizer meaningfully; even directionally better forecasts help
- A true out-of-sample holdout backtest is essential -- comparing frontiers in-sample tells you nothing about real performance
- PyPortfolioOpt is a free, open-source library that implements Markowitz optimization correctly, avoiding the need to hand-code quadratic programming
- This project demonstrates the exact quant research pattern of using ML to improve a classical financial model's inputs, not replace the framework entirely

---

## 🚀 How to Run

1. Open the `.py` file in **Google Colab** (free tier — no GPU or paid compute needed) or run it locally with Python 3.
2. Install dependencies with the `pip install ...` line at the top of the script (all free, open-source packages).
3. (Optional) Download the real dataset from the Kaggle link above and place it in the same folder — the filename the script expects is noted in the code's data-loading step. If you skip this, the script auto-generates sample data so you can still see it run.
4. Run the script top to bottom. Outputs (charts, CSVs, model files) are saved in the working directory.

```bash
pip install -r requirements.txt   # or the pip install line at the top of the script
python MLEng_09_Portfolio_Optimization_Markowitz_ML.py
```

---

## 📂 Repo Structure

```
portfolio-optimization-model-markowitz-+-ml-enhanced/
├── MLEng_09_Portfolio_Optimization_Markowitz_ML.py       # complete, runnable, free-only solution code
├── README.md              # this file
└── outputs/                # charts, CSVs, and model files generated on run
```

---

## ⚠️ Disclaimer

This project is built for educational and portfolio purposes to demonstrate applied ML/quant-risk
skills. It is not financial, credit, or investment advice, and should not be used for real lending,
trading, or compliance decisions without proper review by a licensed professional.

---

*Part of a 20-project AI Engineer + ML Engineer portfolio focused on finance and consulting use cases —
built entirely with free, open-source tools.*
