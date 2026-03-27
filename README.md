# Cross-Sectional Expected Return Model

A **production-grade cross-sectional expected return forecasting pipeline** that evaluates the efficacy of style and custom alpha factors using US Equities, estimates factor premia via Fama-MacBeth two-pass regression, and constructs cost-aware neutralised long/short portfolios.

> **Expected Return = f(Value, Momentum, Quality, Low Volatility, Size, Liquidity, Earnings Surprise Momentum)**

---

## Hypothesis

Style factors (Value, Momentum, Quality, Low Volatility, Size, Liquidity) and short-term earnings revision proxies exhibit **statistically significant, stable premia** in the US equity market and can be synthesised into a cost-aware, tradable long/short model.

---

## Quick Start

```bash
# Clone
git clone https://github.com/<your-username>/cross-sectional-expected-return-model.git
cd cross-sectional-expected-return-model

# Setup
python -m venv venv
source venv/bin/activate        # Unix
.\venv\Scripts\Activate.ps1     # Windows

pip install -r requirements.txt

# Run the full pipeline
python -m src.main

# Use cached data (skip re-downloading)
python -m src.main --skip-download

# Launch the interactive dashboard
streamlit run src/dashboard/app.py

# Run tests
python -m pytest tests/ -v
```

---

## Project Architecture

```
├── src/
│   ├── config.py                 # Central configuration
│   ├── main.py                   # End-to-end pipeline runner
│   ├── data/
│   │   ├── universe.py           # S&P 500 universe builder
│   │   ├── prices.py             # Price download & forward return engine
│   │   └── fundamentals.py       # SimFin / yfinance fundamental data
│   ├── factors/
│   │   ├── style_factors.py      # 6 canonical style factors
│   │   ├── alpha_factor.py       # Custom: Earnings Surprise Momentum (ESM)
│   │   ├── preprocessing.py      # Winsorize → fill → z-score pipeline
│   │   └── orthogonalization.py  # Gram-Schmidt & residualization
│   ├── models/
│   │   ├── cross_sectional.py    # Single-period cross-sectional OLS
│   │   ├── fama_macbeth.py       # Two-pass Fama-MacBeth with Newey-West
│   │   └── expected_return.py    # E[R] forecasts, ranking, decile portfolios
│   ├── portfolio/
│   │   ├── construction.py       # Market/sector/beta neutral portfolios
│   │   ├── costs.py              # Transaction cost & capacity model
│   │   └── backtest.py           # Walk-forward backtesting engine
│   ├── analytics/
│   │   ├── metrics.py            # IC, Sharpe, Sortino, MDD, alpha/beta
│   │   └── visualizations.py     # Research-grade plots (IB aesthetic)
│   └── dashboard/
│       ├── app.py                # Streamlit dashboard (Matrix themed)
│       └── matrix_theme.py       # CSS, Plotly template, digital rain
├── tests/
│   ├── test_data.py
│   ├── test_factors.py
│   ├── test_regression.py
│   └── test_portfolio.py
├── reports/
│   ├── research_paper.md         # Full research paper
│   ├── research_journal.md       # Daily research log
│   ├── experiment_tracker.csv    # Experiment tracking
│   └── figures/                  # Generated plots
├── data/                         # Auto-generated caches (gitignored)
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## Methodology

### Data
- **Universe**: S&P 500 constituents (Yahoo Finance for prices, SimFin for fundamentals)
- **Period**: 2010-01-01 to 2023-12-31
- **Survivorship bias**: Mitigated via broad constituent lists (see Limitations)
- **Look-ahead bias**: Fundamentals lagged by 90 days from report date

### Factor Construction

| Factor | Signal | Construction |
|--------|--------|-------------|
| Value | B/P, E/P, EBITDA/EV composite | Higher = cheaper |
| Momentum | 12-1 month return | Skip recent month (reversal) |
| Size | −log(market cap) | Small cap premium |
| Low Vol | −60-day realised σ | Low vol anomaly |
| Quality | ROE + inv(D/E) + earnings stability | Composite score |
| Liquidity | −avg daily dollar volume | Illiquidity premium |
| **ESM** ★ | Standardised Unexpected Earnings | Custom alpha signal |

### Statistical Framework
- **Fama-MacBeth (1973)** two-pass regression with **Newey-West (1987)** HAC standard errors
- Cross-sectional z-score preprocessing with ±3σ clipping
- Gram-Schmidt sequential orthogonalisation

### Portfolio Construction
- Long/short decile portfolio (D10 long, D1 short)
- Market, sector, and beta neutralisation
- 10 bps one-way transaction costs
- Walk-forward out-of-sample validation

---

## Bias Controls

| Bias | Mitigation |
|------|-----------|
| Survivorship | Broad constituent list (documented limitation) |
| Look-ahead | 90-day fundamental lag, point-in-time alignment |
| Data snooping | In-sample / out-of-sample split, walk-forward | 
| Outliers | Winsorisation at 1st/99th percentile |
| Multicollinearity | Gram-Schmidt orthogonalisation |

---

## Key Outputs

After running the pipeline:

| Artefact | Location |
|----------|----------|
| Factor premia table | `reports/factor_premia_summary.csv` |
| IC series | `reports/ic_series.csv` |
| Gross performance | `reports/gross_performance.csv` |
| Net performance | `reports/net_performance.csv` |
| Rolling premia plot | `reports/figures/rolling_factor_premia.png` |
| Correlation heatmaps | `reports/figures/corr_before.png`, `corr_after.png` |
| Decile performance | `reports/figures/decile_performance.png` |
| Drawdown chart | `reports/figures/drawdown.png` |
| Rolling t-stat | `reports/figures/rolling_t_stat.png` |
| Regime comparison | `reports/figures/regime_comparison.png` |

---

## Failure Cases & Limitations

1. Free data does not provide true point-in-time index constituents (survivorship bias remains approximate)
2. Fundamental lag is assumed at 90 days; actual filing dates vary by company
3. Linear cost model — no market impact or slippage modelling
4. Results specific to US large-cap; may not generalise
5. Linear factor model — no interaction or regime-switching effects

---

## Future Extensions

- Bayesian shrinkage expected return model
- Machine learning cross-sectional model (XGBoost, neural net) with time-series CV
- Dynamic factor weighting via macro regime indicators
- International universe extension (MSCI EM, Japan, Europe)

---

## Interview Translation

**Q: Why does Fama-MacBeth work?**
A: It handles the cross-sectional correlation in residuals that panel OLS ignores. By running independent cross-sections at each date, then computing time-series means, it avoids downward-biased standard errors.

**Q: What causes factor instability?**
A: Crowding (too many investors trading the same signal), regime changes (e.g., value in low-rate vs. high-rate environments), and structural market shifts (e.g., growth of passive investing).

**Q: Risk model vs. alpha model?**
A: A risk model explains the variance of returns (defensive: "what could go wrong"). An alpha model forecasts the mean of returns (offensive: "where is the opportunity"). This project builds an alpha model using risk-model-style factor decomposition.

---

## License

MIT
