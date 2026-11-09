## Primary test — unseen species, genus-imputed stratum (ΔR² = image − genus imputation)
| trait | n | R² imputation | R² image | ΔR² [95% CI] | MDE (ρ .3/.5/.7) | incremental ΔR² [95% CI] |
|---|---|---|---|---|---|---|
| Height | 37 | 0.895 | 0.717 | -0.178 [-8.437, +0.036] | 0.02/0.02/0.015 | -0.061 [-3.288, +0.039] |
| LeafArea | 37 | 0.576 | 0.556 | -0.019 [-0.560, +0.463] | 0.05/0.05/0.045 | +0.055 [-0.563, +0.492] |
| SLA | 37 | 0.251 | 0.283 | +0.033 [-0.716, +0.454] | 0.09/0.085/0.07 | +0.079 [-0.336, +0.348] |
| Leaf_N | 37 | 0.280 | 0.122 | -0.158 [-0.851, +0.260] | 0.07/0.065/0.06 | +0.019 [-0.455, +0.281] |

## Strata within unseen-species CV (ΔR² image − imputation)
| trait | stratum | n | R² imp | R² img | ΔR² [CI] |
|---|---|---|---|---|---|
| Height | genus_imputed | 37 | 0.895 | 0.717 | -0.178 [-8.437, +0.036] |
| Height | family_imputed | 134 | 0.192 | 0.774 | +0.582 [+0.410, +0.775] |
| Height | order_or_global | 66 | 0.110 | 0.740 | +0.630 [+0.427, +0.901] |
| Height | pooled | 237 | 0.306 | 0.767 | +0.460 [+0.328, +0.634] |
| LeafArea | genus_imputed | 37 | 0.576 | 0.556 | -0.019 [-0.560, +0.463] |
| LeafArea | family_imputed | 134 | -0.096 | 0.313 | +0.409 [+0.213, +0.639] |
| LeafArea | order_or_global | 66 | -0.137 | 0.410 | +0.547 [+0.403, +0.749] |
| LeafArea | pooled | 237 | 0.007 | 0.381 | +0.374 [+0.235, +0.558] |
| SLA | genus_imputed | 37 | 0.251 | 0.283 | +0.033 [-0.716, +0.454] |
| SLA | family_imputed | 134 | -0.271 | 0.277 | +0.548 [+0.355, +0.774] |
| SLA | order_or_global | 66 | -0.086 | 0.365 | +0.450 [+0.259, +0.713] |
| SLA | pooled | 237 | -0.139 | 0.307 | +0.445 [+0.312, +0.620] |
| Leaf_N | genus_imputed | 37 | 0.280 | 0.122 | -0.158 [-0.851, +0.260] |
| Leaf_N | family_imputed | 134 | 0.097 | 0.164 | +0.067 [-0.117, +0.286] |
| Leaf_N | order_or_global | 66 | -0.106 | 0.101 | +0.206 [-0.007, +0.486] |
| Leaf_N | pooled | 237 | 0.059 | 0.149 | +0.090 [-0.032, +0.252] |

## Decay with taxonomic distance (pooled; ΔR² image − imputation; incremental)
| trait | level held out | R² imp | R² img | ΔR² [CI] | incremental ΔR² [CI] |
|---|---|---|---|---|---|
| Height | Species | 0.306 | 0.767 | +0.460 [+0.328, +0.634] | +0.459 [+0.347, +0.602] |
| Height | genus | 0.197 | 0.761 | +0.564 [+0.423, +0.758] | +0.559 [+0.430, +0.734] |
| Height | family | 0.072 | 0.739 | +0.667 [+0.519, +0.863] | +0.671 [+0.546, +0.845] |
| Height | order | -0.047 | 0.757 | +0.804 [+0.759, +0.865] | +0.784 [+0.738, +0.845] |
| LeafArea | Species | 0.007 | 0.381 | +0.374 [+0.235, +0.558] | +0.394 [+0.271, +0.547] |
| LeafArea | genus | -0.108 | 0.330 | +0.438 [+0.307, +0.613] | +0.427 [+0.301, +0.593] |
| LeafArea | family | -0.206 | 0.328 | +0.534 [+0.393, +0.727] | +0.513 [+0.386, +0.682] |
| LeafArea | order | -0.038 | 0.318 | +0.356 [+0.278, +0.438] | +0.339 [+0.267, +0.415] |
| SLA | Species | -0.139 | 0.307 | +0.445 [+0.312, +0.620] | +0.416 [+0.300, +0.567] |
| SLA | genus | -0.174 | 0.289 | +0.463 [+0.287, +0.680] | +0.463 [+0.295, +0.663] |
| SLA | family | -0.010 | 0.248 | +0.258 [+0.163, +0.372] | +0.251 [+0.159, +0.361] |
| SLA | order | -0.024 | 0.243 | +0.266 [+0.204, +0.333] | +0.248 [+0.180, +0.315] |
| Leaf_N | Species | 0.059 | 0.149 | +0.090 [-0.032, +0.252] | +0.195 [+0.109, +0.311] |
| Leaf_N | genus | 0.088 | 0.150 | +0.062 [-0.078, +0.229] | +0.148 [+0.043, +0.282] |
| Leaf_N | family | -0.052 | 0.069 | +0.120 [+0.004, +0.232] | +0.058 [-0.068, +0.182] |
| Leaf_N | order | -0.031 | 0.128 | +0.159 [+0.085, +0.222] | +0.153 [+0.086, +0.210] |

## Organ control (genus-imputed stratum)
| trait | weighted − unweighted [CI] | top-10 − unweighted [CI] |
|---|---|---|
| Height | -0.053 [-1.690, +0.054] | -0.045 [-1.230, +0.021] |
| LeafArea | +0.016 [-0.015, +0.120] | -0.005 [-0.139, +0.044] |
| SLA | +0.015 [-0.064, +0.127] | -0.009 [-0.108, +0.090] |
| Leaf_N | +0.052 [-0.103, +0.342] | +0.034 [-0.167, +0.379] |

## Warm-up (seen species, image level)
| trait | R² lookup (nearest species centroid) | R² direct regression |
|---|---|---|
| Height | 0.454 | 0.625 |
| LeafArea | -0.080 | 0.282 |
| SLA | -0.179 | 0.230 |
| Leaf_N | -0.320 | 0.120 |

## Selection-bias checks
| trait | Spearman(dead rate, log trait) | sensitivity: low-dead genus-imputed n | ΔR² [CI] |
|---|---|---|---|
| Height | +0.162 | 22 | -0.115 [-4.516, +0.051] |
| LeafArea | -0.002 | 22 | +0.212 [-0.136, +0.915] |
| SLA | +0.012 | 22 | -0.041 [-0.611, +0.639] |
| Leaf_N | +0.046 | 22 | -0.319 [-0.978, -0.076] |

## Internal checks (Holm on one-sided bootstrap p; triggers only)
```
{
 "Height": {
  "holm_threshold": 0.05,
  "p_boot": 0.96,
  "triggers_checks": false
 },
 "LeafArea": {
  "holm_threshold": 0.016666666666666666,
  "p_boot": 0.54,
  "triggers_checks": false
 },
 "Leaf_N": {
  "holm_threshold": 0.025,
  "p_boot": 0.844,
  "triggers_checks": false
 },
 "SLA": {
  "holm_threshold": 0.0125,
  "p_boot": 0.402,
  "triggers_checks": false
 }
}
```
n_species=237, n_images=7079, imputation levels (Species CV): {'family': 134, 'genus': 37, 'global': 18, 'order': 48}