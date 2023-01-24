# SurvivalGAN: Generating Time-to-Event Data for Survival Analysis
 
This repository contains the experimental code of SurvivalGAN, a generative model that handles survival data firstly by addressing the imbalance in the censoring and time horizons, and secondly by using a dedicated mechanism for approximating time-to-event/censoring. For more details, please read our AISTATS 2023 paper: 'SurvivalGAN: Generating time-to-event Data for Survival Analysis'. 
 
The implementation of the method is included in the [synthcity library](https://github.com/vanderschaarlab/synthcity), in the [SurvivalGAN plugin](https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/plugins/              survival_analysis/plugin_survival_gan.py).
 
# 1. Installation
 
Install `synthcity` in the developer mode
```bash
pip install synthcity[testing]
```
