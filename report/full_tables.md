## Primary test — unseen species, genus-imputed stratum (ΔR² = image − genus imputation)
| trait | n | R² imputation | R² image | ΔR² [95% CI] | MDE (ρ .3/.5/.7) | incremental ΔR² [95% CI] |
|---|---|---|---|---|---|---|
| Height | 1480 | 0.880 | 0.763 | -0.117 [-0.148, -0.089] | 0.02/0.02/0.015 | +0.023 [+0.014, +0.034] |
| LeafArea | 1480 | 0.664 | 0.439 | -0.225 [-0.295, -0.160] | 0.05/0.05/0.045 | +0.066 [+0.048, +0.086] |
| SLA | 1480 | 0.416 | 0.350 | -0.066 [-0.126, -0.005] | 0.09/0.085/0.07 | +0.108 [+0.085, +0.133] |
| Leaf_N | 1480 | 0.548 | 0.167 | -0.381 [-0.447, -0.312] | 0.07/0.065/0.06 | +0.026 [+0.015, +0.038] |

## Strata within unseen-species CV (ΔR² image − imputation)
| trait | stratum | n | R² imp | R² img | ΔR² [CI] |
|---|---|---|---|---|---|
| Height | genus_imputed | 1480 | 0.880 | 0.763 | -0.117 [-0.148, -0.089] |
| Height | family_imputed | 767 | 0.558 | 0.784 | +0.225 [+0.175, +0.278] |
| Height | order_or_global | 98 | 0.148 | 0.701 | +0.553 [+0.357, +0.844] |
| Height | pooled | 2345 | 0.744 | 0.769 | +0.025 [-0.008, +0.061] |
| LeafArea | genus_imputed | 1480 | 0.664 | 0.439 | -0.225 [-0.295, -0.160] |
| LeafArea | family_imputed | 767 | 0.116 | 0.389 | +0.273 [+0.196, +0.356] |
| LeafArea | order_or_global | 98 | 0.166 | 0.311 | +0.145 [-0.039, +0.344] |
| LeafArea | pooled | 2345 | 0.454 | 0.422 | -0.032 [-0.089, +0.026] |
| SLA | genus_imputed | 1480 | 0.416 | 0.350 | -0.066 [-0.126, -0.005] |
| SLA | family_imputed | 767 | 0.101 | 0.296 | +0.195 [+0.131, +0.267] |
| SLA | order_or_global | 98 | 0.124 | 0.221 | +0.097 [-0.143, +0.328] |
| SLA | pooled | 2345 | 0.297 | 0.328 | +0.030 [-0.014, +0.079] |
| Leaf_N | genus_imputed | 1480 | 0.548 | 0.167 | -0.381 [-0.447, -0.312] |
| Leaf_N | family_imputed | 767 | 0.222 | 0.135 | -0.087 [-0.161, -0.007] |
| Leaf_N | order_or_global | 98 | 0.099 | 0.020 | -0.079 [-0.315, +0.166] |
| Leaf_N | pooled | 2345 | 0.412 | 0.154 | -0.257 [-0.310, -0.205] |

## Decay with taxonomic distance (pooled; ΔR² image − imputation; incremental)
| trait | level held out | R² imp | R² img | ΔR² [CI] | incremental ΔR² [CI] |
|---|---|---|---|---|---|
| Height | Species | 0.744 | 0.769 | +0.025 [-0.008, +0.061] | +0.096 [+0.081, +0.115] |
| Height | genus | 0.419 | 0.757 | +0.337 [+0.236, +0.452] | +0.351 [+0.273, +0.441] |
| Height | family | 0.046 | 0.756 | +0.709 [+0.567, +0.876] | +0.657 [+0.534, +0.799] |
| Height | order | -0.075 | 0.744 | +0.819 [+0.786, +0.858] | +0.809 [+0.777, +0.849] |
| LeafArea | Species | 0.454 | 0.422 | -0.032 [-0.089, +0.026] | +0.138 [+0.117, +0.162] |
| LeafArea | genus | 0.069 | 0.402 | +0.333 [+0.247, +0.433] | +0.357 [+0.285, +0.441] |
| LeafArea | family | -0.152 | 0.375 | +0.528 [+0.409, +0.659] | +0.561 [+0.453, +0.684] |
| LeafArea | order | -0.047 | 0.378 | +0.424 [+0.388, +0.461] | +0.386 [+0.343, +0.427] |
| SLA | Species | 0.297 | 0.328 | +0.030 [-0.014, +0.079] | +0.137 [+0.119, +0.159] |
| SLA | genus | 0.014 | 0.316 | +0.302 [+0.227, +0.380] | +0.322 [+0.257, +0.391] |
| SLA | family | -0.134 | 0.300 | +0.434 [+0.358, +0.523] | +0.420 [+0.354, +0.500] |
| SLA | order | -0.021 | 0.314 | +0.335 [+0.304, +0.366] | +0.331 [+0.300, +0.362] |
| Leaf_N | Species | 0.412 | 0.154 | -0.257 [-0.310, -0.205] | +0.044 [+0.033, +0.055] |
| Leaf_N | genus | 0.198 | 0.136 | -0.062 [-0.134, +0.020] | +0.112 [+0.071, +0.158] |
| Leaf_N | family | -0.140 | 0.092 | +0.232 [+0.144, +0.326] | +0.228 [+0.150, +0.309] |
| Leaf_N | order | -0.052 | 0.073 | +0.126 [+0.092, +0.158] | +0.147 [+0.114, +0.180] |

## Organ control (genus-imputed stratum)
| trait | weighted − unweighted [CI] | top-10 − unweighted [CI] |
|---|---|---|
| Height | -0.010 [-0.021, +0.001] | -0.014 [-0.026, -0.001] |
| LeafArea | +0.029 [+0.014, +0.045] | +0.024 [+0.007, +0.042] |
| SLA | +0.016 [+0.005, +0.026] | +0.013 [+0.001, +0.025] |
| Leaf_N | +0.010 [+0.001, +0.019] | +0.007 [-0.004, +0.018] |

## Warm-up (seen species, image level)
| trait | R² lookup (nearest species centroid) | R² direct regression |
|---|---|---|
| Height | 0.378 | 0.616 |
| LeafArea | -0.215 | 0.305 |
| SLA | -0.335 | 0.229 |
| Leaf_N | -0.574 | 0.100 |

## Selection-bias checks
| trait | Spearman(dead rate, log trait) | sensitivity: low-dead genus-imputed n | ΔR² [CI] |
|---|---|---|---|
| Height | +0.192 | 871 | -0.107 [-0.135, -0.081] |
| LeafArea | +0.147 | 871 | -0.227 [-0.307, -0.148] |
| SLA | -0.061 | 871 | -0.067 [-0.142, +0.008] |
| Leaf_N | -0.068 | 871 | -0.374 [-0.454, -0.283] |

## Internal checks (Holm on one-sided bootstrap p; triggers only)
```
{
 "Height": {
  "holm_threshold": 0.016666666666666666,
  "p_boot": 1.0,
  "triggers_checks": false
 },
 "LeafArea": {
  "holm_threshold": 0.025,
  "p_boot": 1.0,
  "triggers_checks": false
 },
 "Leaf_N": {
  "holm_threshold": 0.05,
  "p_boot": 1.0,
  "triggers_checks": false
 },
 "SLA": {
  "holm_threshold": 0.0125,
  "p_boot": 0.983,
  "triggers_checks": false
 }
}
```
n_species=2345, n_images=69952, imputation levels (Species CV): {'family': 767, 'genus': 1480, 'global': 16, 'order': 82}