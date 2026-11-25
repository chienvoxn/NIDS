# NIDS Official Model Selection

## Primary model

- Architecture: Multi-Scale Residual CNN + MHSA
- Variant: cnn_mhsa
- SE: disabled
- MHSA: enabled
- Deployment seed: 42
- CIC features: 60
- CIC classes: 15
- Parameters: 40624

## CIC experimental scope

- Original planned seeds: 42, 43, 44
- Completed seeds: 42, 43
- Seed 44: omitted from final course-project scope
- Completed CNN training runs: 8
- All four CNN variants are retained for ablation.

## Official CIC results

CNN + MHSA, mean of two seeds:

| Metric | Value |
|---|---:|
| Accuracy | 99.3026% |
| Weighted Precision | 99.2801% |
| Weighted Recall | 99.3026% |
| Weighted F1 | 99.2559% |
| FPR | 0.3539% |
| F1 sample SD | 0.5893 percentage points |

## Deployment checkpoint

outputs/final/cic_final_runs/seed_42/cnn_mhsa.pt

SHA256:

1a08c6242942aca3680849412c2d6c63c4e9e276752addf2f6699526817fecfb

## Selection rationale

CNN + MHSA was selected after the ablation study.
It achieved higher mean F1 and lower mean FPR than
CNN + SE + MHSA on the CIC dataset.

The architecture remains consistent with the
extended-CNN course project.

The deployment checkpoint uses seed 42, selected
using validation results and integration checks.

## CSV replay evidence

| Scenario | Accuracy | Detection rate | FPR |
|---|---:|---:|---:|
| PortScan | 99.5% | 100% | 1% |
| DoS Hulk | 99.5% | 100% | 1% |
| Bot | 68.5% | 38% | 1% |

These results are from offline CSV replay, not live
network attack experiments.

## Limitations

- Only two CIC random seeds were completed.
- CNN + MHSA has not surpassed the strongest ML
  baselines in the reported weighted F1 results.
- Minority attack classes remain challenging.
- Botnet detection requires additional graph and
  beaconing analysis.
- CTU deployment artifacts require restoration.
- Dashboard acceptance testing is deferred.
- Live feature parity has not been established.
- Verified live AI detection remains disabled.

Historical CNN + SE + MHSA results are preserved
for comparison and reproducibility.
