# SurvivalGAN: Generating Time-to-Event Data for Survival Analysis
 
This repository contains the experimental code of SurvivalGAN, a generative model that handles survival data firstly by addressing the imbalance in the censoring and time horizons, and secondly by using a dedicated mechanism for approximating time-to-event/censoring. For more details, please read our AISTATS 2023 paper: 'SurvivalGAN: Generating time-to-event Data for Survival Analysis'. 
 
The implementation of the method is included in the [synthcity library](https://github.com/vanderschaarlab/synthcity), in the [SurvivalGAN plugin](https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/plugins/survival_analysis/plugin_survival_gan.py).
 
## Installation
 
Install `synthcity` and other depends
```bash
pip install -r requirements.txt
```

For more tutorials and examples, checkout the [Synthcity tutorials section](https://github.com/vanderschaarlab/synthcity#-tutorials).

## Datasets

Add the data in the `experiments/data` folder.

| Dataset               | No. instances | No. censored instances | No. features | Access |
|---------------------------------|------------------------|---------------------------------|-----------------------|---------------------------|
| ACTG 320 clinical trial dataset | 1151                   | 1055                            | 11                    | [Link](https://github.com/sebp/scikit-survival/blob/master/sksurv/datasets/data/actg320.arff)                      |
| METABRIC                        | 1093                   | 609                             | 689                   | [Link](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric)                  |
| CUTRACT                         | 10086                  | 8881                            | 6                     | private                   |
| PHEART                          | 40409                  | 25664                           | 29                    | private                   |
| SEER prostate cancer            | 171942                 | 167568                          | 6                     | private                      |

## Reproducing results

| **Result**        | **Source notebook**                                                                                                    |
|-------------------|------------------------------------------------------------------------------------------------------------------------|
| Figure 1          | [experiments_00_km_plots_tte_models](experiments/experiments_00_km_plots_tte_models.ipynb)                             |
| Table 1,2,9,10,15 | [experiments_01_benchmark_synthetic_survival_data](experiments/experiments_01_benchmark_synthetic_survival_data.ipynb) |
| Table 3           | [experiments_02_sources_of_gain_parametric](experiments/experiments_02_sources_of_gain_parametric.ipynb)               |
| Table 11          | [experiments_04_loglikelihood](experiments/experiments_04_loglikelihood.ipynb)                                         |
| Table 12, 13, 14  | [experiments_05_predicting_censoring](experiments/experiments_05_predicting_censoring.ipynb)                           |
| Table 16          | [experiments_03_gmm_test_perf](experiments/experiments_03_gmm_test_perf.ipynb)                                         |
| Figure 4,5,8      | [plots_00_data_fidelity](experiments/plots_00_data_fidelity.ipynb)                                                     |
| Figure 6,7        | [plots_02_benchmark_gain_of_function](experiments/plots_02_benchmark_gain_of_function.ipynb)                           |

## Citing
```TODO```
