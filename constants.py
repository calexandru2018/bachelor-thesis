# Results of the grid searches
target_params = [
    (
        'con_mpi_Painseverity',
        {'alpha': 129, 'fit_intercept': True, 'normalize': False},
        {'colsample_bylevel': 0.9, 'colsample_bynode': 1, 'colsample_bytree': 0.9, 'gamma': 3, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_weight': 20, 'n_estimators': 250, 'subsample': 0.6}
    ),
    (
        'fol_mpi_Painseverity',
        {'alpha': 199, 'fit_intercept': True, 'normalize': False},
        {'colsample_bylevel': 0.9, 'colsample_bynode': 1, 'colsample_bytree': 0.9, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_weight': 20, 'n_estimators': 300, 'subsample': 0.6}
    ),
    (
        'con_mpi_Paininterfer',
        {'alpha': 94, 'fit_intercept': True, 'normalize': False},
        {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_weight': 10, 'n_estimators': 250, 'subsample': 0.6}
    ),
    (
        'fol_mpi_Paininterfer',
    {'alpha': 148, 'fit_intercept': True, 'normalize': False},
        {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 3, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_weight': 20, 'n_estimators': 275, 'subsample': 0.6}
    ),
    (
        'con_EQ5D_index',
        {'alpha': 199, 'fit_intercept': True, 'normalize': False},
        {'colsample_bylevel': 0.9, 'colsample_bynode': 0.9, 'colsample_bytree': 0.9, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_weight': 20, 'n_estimators': 250, 'subsample': 0.6}
    ),
    (
        'fol_EQ5d_index',
    {'alpha': 0.1, 'fit_intercept': True, 'normalize': True},
        {'colsample_bylevel': 0.9, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 20, 'n_estimators': 250, 'subsample': 0.6}
    ),
    (
        'con_EQ_VAS',
        {'alpha': 199, 'fit_intercept': True, 'normalize': False},
        {'colsample_bylevel': 0.9, 'colsample_bynode': 0.9, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_weight': 10, 'n_estimators': 300, 'subsample': 0.6}
    ),
    (
        'fol_EQ_VAS',
        {'alpha': 0.1, 'fit_intercept': True, 'normalize': True},
        {'colsample_bylevel': 0.9, 'colsample_bynode': 0.9, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_weight': 20, 'n_estimators': 250, 'subsample': 0.6}
    ),
    (
        'con_sf36_pcs',
        {'alpha': 78, 'fit_intercept': True, 'normalize': False},
        {'colsample_bylevel': 0.9, 'colsample_bynode': 1, 'colsample_bytree': 0.9, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_weight': 20, 'n_estimators': 325, 'subsample': 0.6}
    ),
    (
        'fol_sf36_pcs',
        {'alpha': 140, 'fit_intercept': True, 'normalize': False},
        {'colsample_bylevel': 0.9, 'colsample_bynode': 0.9, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_weight': 10, 'n_estimators': 300, 'subsample': 0.6}
    ),
    (
        'con_sf36_mcs',
        {'alpha': 199, 'fit_intercept': True, 'normalize': False},
        {'colsample_bylevel': 0.9, 'colsample_bynode': 0.9, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_weight': 20, 'n_estimators': 250, 'subsample': 0.6}
    ),
    (
        'fol_sf36_mcs',
        {'alpha': 0.1, 'fit_intercept': True, 'normalize': True},
        {'colsample_bylevel': 0.9, 'colsample_bynode': 0.9, 'colsample_bytree': 0.9, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_weight': 1, 'n_estimators': 250, 'subsample': 0.6}
    )
]

custom_cols_dict = {
    'Gender_sex': 'is_male',
    'Age': 'age',
    'Outside_europe': 'is_european',
    'days_no_work': 'days_since_last_workday',
    'Dr_vistits': 'doctor_visits_last_5_years',
    'Pain_duration': 'days_pain_duration',
    'SMPERIOD': 'pain_variation',
    'Persist_Pain_dur': 'days_persistent_pain',
    'PRI': 'n_pain_locations',
    'Education_recode': 'complete_education',
}

features_to_drop = [
    'ini_cpaq_AE',
    'ini_cpaq_PW',
    'ini_TSK',
    'days_since_last_workday',
    'ini_LISAT_Life',
    'ini_LISAT_Work',
    'ini_LISAT_Economy',
    'ini_LISAT_Leisure',
    'ini_LISAT_Friends',
    'ini_LISAT_Sexlife',
    'ini_LISAT_ADL',
    'ini_LISAT_Family',
    'ini_LISAT_Partner',
    'ini_LISAT_SomHealth',
    'ini_LISAT_PsyHealth',
    #'ini_sf36_pcs',
    #'ini_sf36_mcs',
    'ini_mpi_AC',
    'ini_mpi_DYS',
    'ini_mpi_IP',
    'ini_sf36_pf',
    'ini_sf36_rp', 
    'ini_sf36_bp',
    'ini_sf36_gh',
    'ini_sf36_vt',
    'ini_sf36_sf',
    'ini_sf36_re',
    'ini_sf36_mh'
]