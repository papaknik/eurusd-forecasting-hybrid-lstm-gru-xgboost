# EUR/USD Forecasting: A Hybrid LSTM-GRU + XGBoost Baseline Univariate Model

This project presents a **baseline univariate forecasting model** designed to predict the next-day EUR/USD exchange rate using only historical closing prices. A **stacked LSTM-GRU neural network** captures temporal dependencies in the data, while an **XGBoost regressor** models the residuals — forming a hybrid architecture.

The notebook includes both static test set evaluation and rolling window backtesting from 2007 to 2024, providing insights into how this hybrid approach generalizes across different market regimes.

---

## 📈 Summary of Results

### 📊 Test Set Comparison

| Model                          | MAE       | MAPE     | Directional Accuracy |
|-------------------------------|-----------|----------|-----------------------|
| **Pure RNN (LSTM + GRU)**     | 0.004551  | 0.42%    | 49.79%                |
| **Hybrid (RNN + XGBoost)**    | 0.002149  | 0.20%    | 65.24%                |

> ✅ **Hybrid Benefits**:
> - MAE reduced by ~53%
> - MAPE cut in half
> - Directional Accuracy increased by **+15.5%**

### 🔁 Rolling Evaluation (Hybrid Model Only)
Performed from 2007 to 2024, each rolling window uses an 8-year training period to forecast the next year using a retrained hybrid model (LSTM-GRU + XGBoost).
This approach simulates a realistic walk-forward strategy that adapts to shifting market regimes.


| Metric                        | Value     |
|-------------------------------|-----------|
| Average Rolling MAE           | 0.001842  |
| Average Rolling MAPE          | 0.17%     |
| Avg. Rolling Directional Accuracy | 76.03% |

---

## 🔍 Findings and Conclusion

This hybrid modeling experiment — combining a stacked LSTM-GRU neural network with an XGBoost regressor on residuals — demonstrated good predictive power on historical data up to 2024.

Despite training on a larger dataset (2007–2024), the hybrid model performed slightly worse on the full test set compared to the rolling window evaluation. This may seem counterintuitive but highlights a key aspect of temporal modeling:

- The full test set spans 467 days, encompassing broader market variability including structural shifts and volatility spikes.
- Each rolling window test spans one year (~252 days), capturing locally consistent patterns that improve short-term predictive accuracy.
- Rolling models benefit from **data recency**, whereas static models may be affected by outdated historical relationships.

This reinforces a core principle in time series forecasting:

> **More data isn’t always better** — especially in finance, where structural breaks are common and dynamic retraining strategies are essential.

---

### ⚠️ Caveats and Limitations

While these results are more than promising—especially for a univariate baseline—several important caveats must be acknowledged:

- **Univariate models trained solely on historical values can fail to generalize under shifting market conditions.**  
- **Exchange rate dynamics are driven by exogenous factors** like central bank policy, geopolitical risks, and sentiment — none of which are captured in this univariate model (except indirectly through lagged effects).

#### This underscores the importance of:

- Continuous validation and performance monitoring  
- Multivariate modeling with external macro and sentiment features  
- Adaptive systems that evolve with market regimes  
- Hybrid ensembles combining diverse modeling strategies and timeframes  

---

## 📁 Project Structure

```
EURUSD_HybridForecasting/
│
├── EUR_USD_Forecasting_Univariate_LSTM.ipynb   # Main notebook with end-to-end modeling and evaluation
├── hybrid_model_utils.py                       # Helper functions for modeling, evaluation, and backtesting
├── environment.yml                             # Conda environment specification for reproducibility
├── eurusd_forecasting.pdf                      # PDF version of the notebook for offline access/review
```

---

This project demonstrates the potential of hybrid modeling even in univariate settings, while reminding us that **forecasting is not just about fitting the past**.
