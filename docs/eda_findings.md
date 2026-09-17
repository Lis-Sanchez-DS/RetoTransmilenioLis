# Pulso TransMi EDA findings

## Dataset

The PostgreSQL table `"Original Data"` contains 51,840 observations from 12
stations at 15-minute intervals. The available period is July 26 through
September 9, 2026. The dataset contains 4,320 observations per station.

The demand variable is a non-negative count. The first day contains 1,152
observations (12 stations × 96 intervals) without a previous-day reference.

## Demand behavior

The original time-series plots show strong recurring intraday patterns. Most
stations have pronounced morning and evening peaks, with lower demand during
the night and between peak periods. Demand levels differ substantially by
station:

- Banderas, Ricaurte - NQS, Portal Américas, and Portal El Dorado generally
  have the largest demand ranges.
- Universidades - CityU generally has the lowest range.
- The peak shape and magnitude vary by station, so a single global baseline
  should be complemented by station-specific features or models.
- Weekdays and weekends have visibly different patterns, indicating weekly
  structure in addition to daily seasonality.

## Deseasonalization

Daily seasonality was removed separately for each station by subtracting the
median demand for the corresponding 15-minute time-of-day slot. There are 96
slots per day. This removes the typical daily profile while preserving
unexpected changes and longer-term effects.

The deseasonalized plots still show multi-interval runs of positive and
negative residuals, especially around unusual demand periods and weekends.
This means daily deseasonalization does not remove all predictable structure.

## Autocorrelation

The ACF was calculated for each station's deseasonalized residuals through lag
96, where lag 96 equals one day. Most stations retain significant positive
autocorrelation over many lags. Several show especially persistent behavior,
including Universidades - CityU and Universidad Nacional. Calle 100 -
Marketmedios and Calle 72 also show longer-range structure.

The residuals therefore cannot be treated as white noise merely because they
fall inside an outlier threshold. The persistence indicates that lagged demand,
rolling statistics, or autoregressive terms may improve forecasts. Weekly
effects should also be represented explicitly, for example with weekday and
weekend features or a weekly seasonal model.

## Outlier analysis

Potential outliers were identified on the deseasonalized residuals using a
station-specific robust MAD rule:

```text
robust_z = (residual - station_median) / (1.4826 × station_MAD)
outlier  = abs(robust_z) > 3.5
```

The threshold graph displays upper and lower station-specific limits as dashed
lines and shades the area beyond those limits in gray. A flagged observation
is not automatically an error; it may represent an event, operational change,
or genuine unusual demand.

## Baseline model

The first baseline uses the demand from the same station exactly one day
earlier. Since the sampling interval is 15 minutes, this is a lag of 96
observations. If the previous-day value is unavailable, the prediction is 0.

Accuracy follows the project metric:

```text
Accuracy = 100 × max(0, 1 − WAPE)
```

WAPE is calculated per station and the station accuracies are averaged. The
baseline achieved an overall Accuracy of 75.86%:

| Station | Accuracy |
| --- | ---: |
| Calle 100 - Marketmedios | 67.74% |
| Portal Suba | 80.61% |
| Portal Américas | 80.39% |
| Banderas | 80.97% |
| Portal El Dorado | 79.76% |
| Universidades - CityU | 68.55% |
| Movistar Arena | 76.37% |
| Universidad Nacional | 69.29% |
| Ricaurte - NQS | 80.68% |
| Portal Usme | 79.85% |
| Calle 72 | 69.82% |
| Museo Nacional | 76.23% |

The baseline is a useful reference because it captures daily seasonality, but
its errors show that daily repetition alone is insufficient. The next models
should consider weekday/weekend effects, recent lags, rolling demand, and
possibly station-specific behavior.

## Modeling implications

Holt-Winters is a suitable interpretable benchmark for level, trend, and daily
seasonality. ARIMA or SARIMA can model the remaining autocorrelation; a daily
seasonal period is 96 intervals and a weekly period is 672 intervals. Because
the history contains only about 45 days, a full 672-point seasonal model
should be evaluated carefully. Feature-based models with lagged demand and
calendar variables are another practical candidate.
