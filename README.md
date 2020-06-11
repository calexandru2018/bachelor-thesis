# Predicting Multimodal Rehabilitation Outcomes using Machine Learning

### Abstract
Chronic pain is a complex health issue and a major cause of disability worldwide. Although multimodal rehabilitation (MMR) has been recognized as an effective form for treatment of chronic pain, some patients do not benefit from it. If treatment outcomes could be reliably predicted, then patients who would benefit more from MMR could be prioritized over others. Since it has been shown that machine learning can accurately predict outcomes in other health-related domains, this study aims to investigate the use of it to predict outcomes of MMR, using data from the Swedish Quality Registry for Pain Rehabilitation (SQRP). XGBoost regression was used for this purpose, and its predictive performance was compared to Ridge regression. 12 models were trained on SQRP data for each algorithm, in order to predict pain and quality of life related outcomes. The results show similar performances for both algorithms, with mean cross-validated R² values of 0.323 and 0.321 for the XGBoost and Ridge models respectively. The mean RMSE values of 6.744 for XGBoost and 6.743 for Ridge were similar as well. Since XGBoost performed similarly to a less computationally expensive method, the use of this method for MMR outcome prediction was not supported by the results of this study. However, machine learning has the potential to be more effective for this purpose, through the use of different hyperparameter values, correlation-based feature selection or other machine learning algorithms.

### Authors
[Alexandru Cheltuitor](https://github.com/calexandru2018) and [Niklas Jones-Quartey](https://github.com/njqdev)

### Requirment
- Python >= v3.6.7

### Dependecies
All development was done within a conda virtual environment. The following dependencies need to be installed:
- [Pytz >= v2020.01](https://anaconda.org/anaconda/pytz)
- [XGBoost >= v1.0.2](https://anaconda.org/conda-forge/xgboost)
- [iPython >= v7.13.0](https://anaconda.org/anaconda/ipython)
- [Seaborn >= v0.10.1](https://anaconda.org/anaconda/seaborn)
- [Matplotlib >= v3.1.3](https://anaconda.org/anaconda/matplotlib)
- [Pyreadstat >= v0.3.4](https://anaconda.org/conda-forge/pyreadstat)
- [SciKit-Learn >= v0.22.1](https://anaconda.org/anaconda/scikit-learn)
- [Statsmodels >= v0.11.1](https://anaconda.org/anaconda/statsmodels)
- [Jupyter Notebook >= v1.0.0](https://anaconda.org/anaconda/jupyter)