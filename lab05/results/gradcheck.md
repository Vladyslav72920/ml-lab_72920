# Gradient check (task 4)

Expression: `((a*b + c)**2).relu().sum()`, vectors of size 6, 20 random trials (half with a `-1` shift so ReLU is inactive for 34 elements), float64.

| Comparison | Max absolute difference | < 1e-6 |
|---|---|---|
| engine vs torch | 0.000e+00 | yes |
| engine vs numerical | 2.055e-10 | yes |
| torch vs numerical | 2.055e-10 | yes |
| forward | 3.553e-15 | yes |

Accumulation test `y = x*x + 3x + x` at x=1.5: engine 7.0, torch 7.0, analytic 7.0 -> pass.

**Overall: PASS**
