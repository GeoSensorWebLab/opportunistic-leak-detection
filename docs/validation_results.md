# Validation Results

Results from synthetic twin experiments and sensitivity analysis.

---

## Strategy Comparison

Strategies are compared across 7 scenarios (A–G) using the following metrics:

| Metric | Description |
|--------|-------------|
| Entropy Reduction % | Fraction of initial uncertainty eliminated |
| Localization RMSE (m) | Distance from belief peaks to true sources |
| Source Detection Rate | Fraction of true sources detected (belief > 0.3 within 50m) |
| Information Efficiency | Bits of entropy reduced per meter traveled |

### Single-Seed Results

*Run with:* `uv run python experiments/run_strategy_comparison.py --steps 20`

| Strategy | Avg Ent % | Avg RMSE | Avg Det Rate | Avg Efficiency |
|----------|-----------|----------|--------------|----------------|
| EER | — | — | — | — |
| Opportunistic | — | — | — | — |
| Max Detection | — | — | — | — |
| Grid Search | — | — | — | — |
| Random | — | — | — | — |

### Multi-Seed Results (Variance Reporting)

*Run with:* `uv run python experiments/run_strategy_comparison.py --seeds 42 123 456 --steps 20`

| Strategy | Ent % (mean ± std) | RMSE (mean ± std) | Det Rate (mean ± std) |
|----------|--------------------|--------------------|----------------------|
| EER | — | — | — |
| Opportunistic | — | — | — |
| Max Detection | — | — | — |
| Grid Search | — | — | — |
| Random | — | — | — |

---

## Sensitivity Analysis

One-at-a-time parameter sweep using Scenario A with the EER strategy.

*Run with:* `uv run python experiments/run_sensitivity_analysis.py --steps 20`

### Detection Threshold (ppm)

| Value | Entropy % | RMSE | Det Rate | Efficiency |
|-------|-----------|------|----------|------------|
| 3.0 | — | — | — | — |
| 5.0 | — | — | — | — |
| 7.0 | — | — | — | — |

### Sensor MDL (ppm)

| Value | Entropy % | RMSE | Det Rate | Efficiency |
|-------|-----------|------|----------|------------|
| 0.5 | — | — | — | — |
| 1.0 | — | — | — | — |
| 2.0 | — | — | — | — |

### False Alarm Rate

| Value | Entropy % | RMSE | Det Rate | Efficiency |
|-------|-----------|------|----------|------------|
| 0.001 | — | — | — | — |
| 0.01 | — | — | — | — |
| 0.1 | — | — | — | — |

### Deviation Scale (m)

| Value | Entropy % | RMSE | Det Rate | Efficiency |
|-------|-----------|------|----------|------------|
| 25 | — | — | — | — |
| 50 | — | — | — | — |
| 100 | — | — | — | — |

### EER Subsample Factor

| Value | Entropy % | RMSE | Det Rate | Efficiency |
|-------|-----------|------|----------|------------|
| 1 | — | — | — | — |
| 2 | — | — | — | — |
| 4 | — | — | — | — |
| 8 | — | — | — | — |

### Prior Kernel Radius (m)

| Value | Entropy % | RMSE | Det Rate | Efficiency |
|-------|-----------|------|----------|------------|
| 50 | — | — | — | — |
| 100 | — | — | — | — |
| 200 | — | — | — | — |

---

## Summary

### Key Findings

1. **Strategy ranking**: *(to be filled after running experiments)*
2. **Most sensitive parameters**: *(to be filled after running sensitivity analysis)*
3. **Robust parameter ranges**: *(to be filled after analysis)*

### Scenarios

| Scenario | Description | Purpose |
|----------|-------------|---------|
| A | Single source, steady wind | Basic detection |
| B | Three sources, steady wind | Multi-source localization |
| C | Single source, variable wind | Wind robustness |
| D | Three sources, variable wind | Full pipeline stress test |
| E | Two intermittent sources | Duty cycle behavior |
| F | No active leaks | Null hypothesis / false alarm rate |
| G | All sources at 10x emission | Saturation stress test |
