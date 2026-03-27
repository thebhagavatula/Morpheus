# Cross-Sectional Expected Return Model: Research Paper

## 1. Economic Motivation

Style factors — Value, Momentum, Quality, Low Volatility, Size, and Liquidity — have been documented extensively in academic literature as persistent sources of excess return in equity markets (Fama & French 1993, Carhart 1997, Asness et al. 2013). This research project constructs a **cross-sectional expected return model** that synthesises these premia into a unified, tradable signal.

The key economic questions are:

1. **Do factor premia remain statistically significant** in the post-publication era (2010–2023)?
2. **Are premia stable** across regimes, or do factors exhibit dormancy?
3. **Can the model produce alpha** after realistic transaction costs?

We extend the canonical six-factor framework with a custom **Earnings Surprise Momentum (ESM)** signal, exploiting the well-documented post-earnings-announcement drift (PEAD) anomaly.

---

## 2. Factor Construction Methodology

### 2.1 Universe

- **Source**: S&P 500 constituents (Wikipedia), approximating a broad US large-cap universe.
- **Survivorship bias**: Mitigated by using broad constituent lists. We document the limitation that free data does not provide true historical constituent snapshots.
- **Period**: January 2010 – December 2023.

### 2.2 Style Factors

| Factor | Construction | Sign Convention |
|--------|-------------|-----------------|
| **Momentum** | 12-month return, skip most recent month (12-1) | Higher = stronger trend |
| **Size** | −log(market cap) | Higher = smaller |
| **Low Volatility** | −annualised 60-day σ | Higher = lower vol |
| **Liquidity** | −average daily dollar volume | Higher = more illiquid |
| **Value** | Mean of B/P, E/P, EBITDA/EV | Higher = cheaper |
| **Quality** | 40% ROE + 30% inv(D/E) + 30% earnings stability | Higher = better quality |

### 2.3 Custom Alpha: Earnings Surprise Momentum

We implement Standardised Unexpected Earnings (SUE):

```
SUE_t = (EPS_t − EPS_{t−4}) / σ(EPS_t − EPS_{t−4})
```

When time-series EPS is unavailable, we fall back to:

```
ESM = (Forward EPS / Trailing EPS) − 1
```

### 2.4 Preprocessing

All factors undergo cross-sectional preprocessing at each date:
1. **Winsorization** at the 1st and 99th percentiles
2. **Missing data treatment**: drop tickers with >50% NaN; fill remainder with sector-median
3. **Z-score normalisation**: mean 0, std 1, clipped at ±3σ

### 2.5 Orthogonalisation

Sequential Gram-Schmidt orthogonalisation removes multicollinearity. Priority order reflects economic conviction: Momentum → Low Vol → Liquidity → ESM.

---

## 3. Statistical Framework

### 3.1 Cross-Sectional Regression

At each month *t*:

```
R_{i,t} = α_t + Σ_k λ_{k,t} · F_{ik,t} + ε_{i,t}
```

### 3.2 Fama-MacBeth Two-Pass

- **Pass 1**: Run the cross-sectional regression at each *t* → λ̂_{k,t}
- **Pass 2**: Compute time-series mean of λ̂ and test significance:
  - OLS standard errors: SE = σ_λ / √T
  - **Newey-West HAC** (6 lags, Bartlett kernel): corrects for serial correlation

### 3.3 Expected Return Forecast

```
E[R_i] = Σ_k  λ̄_k · F_{ik}
```

where λ̄_k is a 12-month rolling average of estimated premia.

---

## 4. Empirical Results

*(Generated dynamically by the pipeline — see `reports/` directory after execution)*

Key outputs:
- **Factor premia summary table**: mean λ, t-stats, NW t-stats (`reports/factor_premia_summary.csv`)
- **IC analysis**: mean IC, IC IR, % positive (`reports/ic_series.csv`)
- **Performance metrics**: Sharpe, Sortino, MDD, alpha, beta (`reports/gross_performance.csv`, `reports/net_performance.csv`)

### Visualisations

- Rolling factor premia (`reports/figures/rolling_factor_premia.png`)
- Factor correlation before/after orthogonalisation (`reports/figures/corr_before.png`, `corr_after.png`)
- Decile performance spread (`reports/figures/decile_performance.png`)
- Drawdown chart (`reports/figures/drawdown.png`)
- Rolling t-statistic with significance bands (`reports/figures/rolling_t_stat.png`)
- Crisis vs. normal regime comparison (`reports/figures/regime_comparison.png`)

---

## 5. Tradeability Analysis

### 5.1 Transaction Costs

A linear cost model of 10 bps per side is applied. Turnover is measured as one-way weight change per rebalance.

### 5.2 Capacity

Estimated using:
```
Max AUM ≈ (Participation Rate × mean ADV × N_stocks) / mean Turnover
```

### 5.3 Portfolio Neutralisation

| Constraint | Method |
|-----------|--------|
| Market neutral | Dollar-neutral (Σw = 0) |
| Sector neutral | Equal long/short within each GICS sector |
| Beta neutral | Adjust weights to achieve portfolio β = 0 |

---

## 6. Limitations

1. **Survivorship bias**: Free data provides only the *current* S&P 500 list, not historical constituents. True point-in-time membership requires CRSP.
2. **Fundamental data lag**: We assume a 90-day reporting delay; actual filing dates vary.
3. **No intraday or market microstructure modelling**: Our cost model is simplified.
4. **Single universe**: Results are specific to US large-cap equities and may not generalise.
5. **Linear model**: Non-linear factor interactions and regime-dependent premia are not captured.

---

## 7. Future Research

- **Bayesian shrinkage** of factor premia for improved out-of-sample forecasts
- **Machine learning** cross-sectional model (gradient boosting, neural net) with proper time-series CV
- **Dynamic factor weighting** based on macro regime indicators
- **International extension**: apply to MSCI EM, Japan, Europe universes
- **Intraday alpha decay** study using tick-level data

---

## References

- Fama, E.F. & MacBeth, J.D. (1973). *Risk, Return, and Equilibrium: Empirical Tests*. Journal of Political Economy.
- Fama, E.F. & French, K.R. (1993). *Common Risk Factors in the Returns on Stocks and Bonds*. Journal of Financial Economics.
- Carhart, M.M. (1997). *On Persistence in Mutual Fund Performance*. Journal of Finance.
- Asness, C.S., Moskowitz, T.J. & Pedersen, L.H. (2013). *Value and Momentum Everywhere*. Journal of Finance.
- Newey, W.K. & West, K.D. (1987). *A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*. Econometrica.
