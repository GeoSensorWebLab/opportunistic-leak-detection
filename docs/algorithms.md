# Algorithm Documentation

Mathematical formulations for the Methane Leak Opportunistic Tasking System.

---

## 1. Gaussian Plume Dispersion

### 1.1 Standard (Instantaneous) Plume

Concentration at receptor point (x, y, z) from a continuous point source at (x_s, y_s, H):

```
C(x,y,z) = Q / (2π u σ_y σ_z)
            × exp(-½ (y_c / σ_y)²)
            × [exp(-½ ((z-H)/σ_z)²) + exp(-½ ((z+H)/σ_z)²)]
```

where:
- **Q** = emission rate (kg/s)
- **u** = wind speed (m/s)
- **σ_y, σ_z** = lateral and vertical dispersion parameters (m)
- **y_c** = crosswind distance from plume centerline (m)
- **H** = source release height (m)
- The second vertical term is the **ground reflection** (image source method)

Coordinate transformation from meteorological wind direction:
- Wind blows FROM direction `d`, so the plume travels TOWARD `(d + 180°) mod 360°`
- Downwind distance: `x_d = Δx·sin(θ) + Δy·cos(θ)` where θ = wind-toward direction
- Crosswind distance: `y_c = -Δx·cos(θ) + Δy·sin(θ)`

### 1.2 Crosswind-Integrated Plume

Integrates the standard plume equation over the crosswind (y) direction, removing σ_y dependence:

```
C_ci(x,z) = Q / (√(2π) u σ_z)
             × [exp(-½ ((z-H)/σ_z)²) + exp(-½ ((z+H)/σ_z)²)]
```

Produces broader, lower-peak plumes that better approximate time-averaged field measurements under turbulent fluctuations.

### 1.3 Gaussian Puff Model

For an instantaneous (episodic) release of total mass M:

```
C_puff = M / ((2π)^(3/2) σ_x σ_y σ_z)
         × exp(-½ ((x_d - u·t) / σ_x)²)
         × exp(-½ (y_c / σ_y)²)
         × [exp(-½ ((z-H)/σ_z)²) + exp(-½ ((z+H)/σ_z)²)]
```

where:
- **M** = total mass released (kg)
- **t** = time since release (s)
- **σ_x = σ_y** (isotropic horizontal dispersion)
- Dispersion parameters evaluated at travel distance `u·t`

---

## 2. Pasquill-Gifford Sigma Parameterization

Power-law form (Turner 1970):

```
σ_y = a_y × x^(b_y)
σ_z = a_z × x^(b_z)
```

where x is downwind distance in meters. Coefficients (a, b) vary by stability class:

| Class | Description      | a_y    | b_y    | a_z    | b_z    |
|-------|------------------|--------|--------|--------|--------|
| A     | Very unstable    | 0.3658 | 0.9031 | 0.192  | 1.2044 |
| B     | Moderately unstable | 0.2751 | 0.9031 | 0.156 | 1.0857 |
| C     | Slightly unstable | 0.2090 | 0.9031 | 0.116 | 0.9865 |
| D     | Neutral          | 0.1471 | 0.9031 | 0.079  | 0.9031 |
| E     | Slightly stable  | 0.1046 | 0.9031 | 0.063  | 0.8314 |
| F     | Very stable      | 0.0722 | 0.9031 | 0.053  | 0.7540 |

Ordering invariant: σ_A > σ_B > σ_C > σ_D > σ_E > σ_F at any fixed distance.

---

## 3. Detection Probability Model

Sigmoid function with a hard Minimum Detection Limit (MDL):

```
P(detect | C) = 0                           if C < MDL
P(detect | C) = 1 / (1 + exp(-k(C - T)))   if C ≥ MDL
```

where:
- **C** = concentration (ppm)
- **MDL** = Minimum Detection Limit (default 1.0 ppm)
- **T** = detection threshold / sigmoid midpoint (default 5.0 ppm, where P = 50%)
- **k** = steepness parameter (default 1.0)

---

## 4. Prior Belief Model (Stage 1)

Per-source prior leak probability combines four factors:

```
P_prior(i) = clip(P_base × F_age × F_prod × F_insp, 0, 1)
```

- **P_base**: Equipment type base rate (from EPA GHGRP/API studies)
- **F_age** = 1 + scale × (age / age_ref)^exponent — superlinear aging
- **F_prod** = 1 + prod_scale × (production / prod_ref) — mechanical stress
- **F_insp** = 1 + exp(-days_since / decay_half_life) — inspection recency

Spatial prior projects point probabilities onto a 2D grid via Gaussian kernels:

```
Prior(x,y) = Σ_i P_prior(i) × exp(-d_i² / (2 × r²))
```

where d_i = distance from (x,y) to source i, and r = kernel radius (default 100m).

---

## 5. Bayesian Update (Stage 2)

Cell-wise Bayes' theorem after a field measurement at location m:

```
P(leak_i | obs) = P(obs | leak_i) × P(leak_i) / P(obs)
```

For a **detection** (obs = +):
```
P(+ | leak_i) = P_detect(C_reverse(i, m))
P(+ | no_leak) = FALSE_ALARM_RATE
```

For a **non-detection** (obs = −):
```
P(− | leak_i) = 1 − P_detect(C_reverse(i, m))
P(− | no_leak) = 1 − FALSE_ALARM_RATE
```

**Reverse plume**: For each grid cell i as a hypothetical source, compute the concentration at measurement point m using the forward Gaussian plume model. This is fully vectorized (no per-cell loop).

Normalization:
```
P(obs) = P(obs | leak_i) × P(leak_i) + P(obs | no_leak) × (1 − P(leak_i))
```

---

## 6. Expected Entropy Reduction (Stage 3)

Binary entropy at each cell:

```
H(p) = −p log₂(p) − (1−p) log₂(1−p)
```

Total map entropy:
```
H_total = Σ_cells H(p_i)
```

For a candidate measurement location m, the Expected Entropy Reduction (EER) is:

```
EER(m) = Σ_cells [H_current(i) − P(obs) × H(posterior_i | obs) − P(¬obs) × H(posterior_i | ¬obs)]
```

where:
- P(obs) = probability of a detection at m under the current belief
- The posterior for each outcome is computed via the Bayesian update formula
- Cells are independent, so total EER is the sum of per-cell reductions

Performance optimization: Uses subsampled grid (EER_SUBSAMPLE = 4) with bilinear interpolation via `RegularGridInterpolator`.

---

## 7. Multi-Source Complementary Probability

Combined detection probability from N independent sources:

```
P_combined = 1 − Π_i (1 − P_i)
```

Fleet coverage with K workers follows the same principle:

```
P_fleet(x,y) = 1 − Π_k (1 − P_worker_k(x,y))
```

---

## 8. Scoring Formula

### Heuristic Mode

```
Score(x,y) = [Prior(x,y) ×] P(detect|x,y) / (PathDeviation(x,y) + ε)
```

where PathDeviation is the minimum distance from (x,y) to the baseline path, ε prevents division by zero.

### Information-Theoretic Mode (EER)

```
Score(x,y) = EER(x,y) / (PathDeviation(x,y) + ε)
```

Only cells within `MAX_DEVIATION_M` of the baseline path are considered. Non-maximum suppression (min 50m separation) ensures spatially diverse recommendations.

### Path Deviation Cost

Exponential decay model:
```
Cost(d) = exp(−d / DEVIATION_SCALE_M)
```

where d is the perpendicular distance from the baseline path. This naturally penalizes large detours while allowing small deviations.

---

## References

- Turner, D.B. (1970). *Workbook of Atmospheric Dispersion Estimates*. EPA.
- Gifford, F.A. (1976). Turbulent diffusion-typing schemes: A review. *Nuclear Safety*, 17(1).
- EPA GHGRP: *Greenhouse Gas Reporting Program*, Subpart W (petroleum and natural gas).
- API (2014): *Compressor and Component Leak Studies*.
