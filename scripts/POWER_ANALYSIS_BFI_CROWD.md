# Power Analysis: BFI Model vs. Crowd of Other Models

Back-of-the-envelope calculations for determining required repeat runs when comparing a single model to a crowd of other AI models on BFI traits.

## Design

- **Model X:** n₁ = number of repeat runs (question-order shuffles)
- **Crowd:** n₂ = N = number of **models** (not total trials)
- **Comparison:** Model X mean (over n₁ runs) vs. mean of the other N models

## Assumptions

- **Effect size:** Medium–large (d ≈ 0.8)
- **α:** 0.01 (two-tailed)
- **Target power:** 80%

## Required Runs (n₁) for α = 0.01, Power = 0.80

| Effect size (d) | n₂ = 25 models | n₂ = 50 models |
|-----------------|----------------|----------------|
| d = 0.8         | 77 runs        | 31 runs        |
| d = 1.0         | 26 runs        | 17 runs        |
| d = 1.2         | 14 runs        | 11 runs        |

**Recommendation:** For medium–large effects (d ≈ 0.8) with α = 0.01, use **~77 runs** per model when crowd has 25 models. For larger effects (d ≈ 1.0), **~26 runs** suffices.

## Ceiling Effect

With n₂ fixed, there is a maximum achievable non-centrality parameter:

```
λ_max = d × √n₂
```

For power 0.80 at α = 0.01, we need λ ≈ 3.4. Thus:

```
d_min = 3.4 / √n₂
```

- **n₂ = 25:** d_min ≈ 0.68 (cannot achieve 80% power for d < 0.68)
- **n₂ = 50:** d_min ≈ 0.48

## Power with Current n₁ = 10

| Effect size | Power (α = 0.01, n₂ = 25) |
|-------------|----------------------------|
| d = 0.8     | 29%                        |
| d = 1.0     | 49%                        |

10 runs is underpowered for medium–large effects at α = 0.01.

## Formula

Two-sample t-test (Welch or pooled), unequal n:

```
λ = d × √(n₁ × n₂ / (n₁ + n₂))
```

Power = P(|T| > t_crit | non-central t with df, λ).
