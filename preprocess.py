import pandas as pd
import numpy as np
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

def impute_data(df, method=False, arg=False):
    """Function that imputes data based on method.
    method:
        - KNN Impute: knn_impute:
          - Default n: 2
        - Simple Impute: simple_impute
        - Multiple/Iterative Impute: iter_impute
          - Default max_iterations: 3
    """
    imputed_df = pd.DataFrame()
    
    if method == "knn":
        # KNN imput
        imp = KNNImputer(n_neighbors=arg if arg else 2)
    elif method == "iter":
        # Multiple/Iterative imput
        imp = IterativeImputer(max_iter=arg if arg else 3, random_state=30)

    elif method == "simple" or not method:
        # Simple Impute test
        imp = SimpleImputer(strategy=arg if arg else "mean")

    imputed_df = pd.DataFrame(imp.fit_transform(df))
    imputed_df.columns = df.columns
    imputed_df.index = df.index 
        
    return imputed_df

def remove_outliers(df, display_values=False):
    df_copy = df.copy()
    
    for col in df_copy.columns:
        std_dev = np.nanstd(df_copy[col])
                   
        if df_copy[col].dtype.name != 'category':
            mean = np.nanmean(df_copy[col])
            
            if display_values:
                print("{} - mean: {} - std: {}".format(col, round(mean, 3), round(std_dev, 3)))
                
            
            df_copy[col] = df_copy[col].apply(lambda x: np.nan if not pd.isnull(x) and (np.abs(x) > (mean+(3*std_dev))) else x)
    
    return df_copy

def drop_empty_rows_for_target_label(orig_df, df, target_tabel):
    """Adds a new column with the target label and drops empty rows for it.
    Returns a tuple with a df pre and post empty row removal and the target label df.
    """
    post_df = pd.DataFrame()
    pre_df = df.copy()
    
    pre_df[target_tabel] = orig_df[target_tabel]
    post_df = pre_df.dropna(subset=[target_tabel]).copy()

    target_label = post_df[target_tabel]
    post_df.drop([target_tabel], axis=1, inplace=True)
    
    return (pre_df, post_df, target_label)

def percentage_diff_target_label(diff_target_name, ini_feature_name, df, orig_df):
    """
    Converts the target diff feature to a percentage one.
    
    Returns ( target_label, imputed_df )
    """
    new_target = 'percent_' + diff_target_name
    
    orig_df[new_target] = orig_df[diff_target_name] / orig_df[ini_feature_name]
    df[new_target] = orig_df[new_target]
    df = df.replace((np.inf, -np.inf), np.nan)
    
    imputed_df = impute_data(df, method="knn")
    
    target_label = imputed_df[new_target]
    imputed_df.drop([new_target], axis=1, inplace=True)
    return target_label, imputed_df
    
def drop_features(df, features_to_drop):
    """Drop unnecessary features
    """
    new_df = df.copy()
    for col in new_df.columns:
        if col in features_to_drop:
            new_df.drop([col], axis=1, inplace=True)

    return new_df


def drop_features_except(df, features_to_keep):
    """Drop all features except the features to keep.
    """
    new_df = df.copy()
    for col in new_df.columns:
        if col not in features_to_keep:
            new_df.drop([col], axis=1, inplace=True)

    return new_df


def drop_LISAT_cols(df):
    """Drop all LISAT columns.
    """
    new_df = df.copy()
    for col in new_df.columns:
        if "LISAT" in col:
            new_df.drop([col], axis=1, inplace=True)

    return new_df

def transform_ordinals(df):
    """Calls transform_LISAT(), transform_NRS7(), transform_is_european() and transform_doctor_visits().
    This transforms existing column values to more understandable ordinal values.
    
    Returns:
        - The same df with all the changes.
    """
    df = transform_LISAT(df)
    df = transform_NRS7(df)
    df["is_european"] = df["is_european"].apply(transform_is_european)
    df["doctor_visits_last_5_years"] = df["doctor_visits_last_5_years"].apply(transform_doctor_visits)
    
    return df

def transform_nominals_to_binary(df):
    """Calls transform_pain_variation() and transform_education().
    This transforms existing column values to more understandable binary values.
    
    Returns:
        - The same df with all the changes.
    """
    df["pain_variation"] = df["pain_variation"].apply(transform_pain_variation)
    df["complete_education"] = df["complete_education"].apply(transform_education)
    
    for s in [df['complete_education'], df['pain_variation']]:
        dummies = pd.get_dummies(s)
        df = pd.concat([df, dummies], axis=1)
        df.drop([s.name], axis=1, inplace=True)
   
    return df

def rename_columns(df, columns_dictionary):
    """Renames provided columns.
    """
    return df.rename(columns=columns_dictionary)

def select_ini_features(df):
    """Select all ini features.
    """
    new_df = pd.DataFrame()
    for col in df.columns:
        if ("ini_" in col or "_ini" in col) and ((not "_fol" in col and not "fol_" in col) and (not "_con" in col and not "con_" in col)):
            new_df[col] = df[col]
            
    return new_df

def insert_base_info(original_df, return_df, custom_cols):
    """Inserts base information columns from original dataframe into the new custom dataframe.
    These base information details are described in custom_cols.
    """
    for col in original_df.columns:
        if any(col in s for s in custom_cols):
            # inserts into (index, col_name, col_value)
            return_df.insert(0, col, original_df[col])
            
    return return_df


def transform_education(row):
    """Recodes education to more understandable strings.
    """
    if row == "Universitet":
        return "university"
    elif row == "grundskola + annat":
        return "elementary_and_other"
    elif row == "gymnasium":
        return "high_school"
    elif row == "Annat":
        return "other"
    else:
        return np.nan

    
def transform_LISAT(df):
    """Recodes LISAT columns to ordered number scale, 1-5.
    """
    def LISAT_to_num(row):
        if row == "Mycket tillfredsställande":
            return 1
        elif row == "Tillfredsställande":
            return 2
        elif row == "Ganska otillfredsställande":
            return 3
        elif row == "Otillfredsställande":
            return 4
        elif row == "Mycket otillfredsställande":
            return 5
        else:
            return np.nan
        
    for col in df.columns:
        if "LISAT" in col:
            df[col] = df[col].apply(LISAT_to_num)
            
    return df

def transform_is_european(row):
    """Binarize is_european.
    """
    if row == "inom Europa":
        return 1
    elif row == "utom Europa":
        return 0
    else:
        return np.nan

def transform_pain_variation(row):
    """Recodes pain_variation to more understandable strings.
    """
    if row == "Ihållande- aldrig smärtfri":
        return "consistent_pain"
    elif row == "Peridodvis återkommande":
        return "periodic_pain"
    else:
        return np.nan
    

def transform_doctor_visits(row):
    """Recodes doctor_visits_last_5_years column to a ordered number scale, 1-3.
    """
    if row == "0-1 ggr":
        return 1
    elif row == "2-3 ggr":
        return 2
    elif row == "4 eller fler":
        return 3
    else:
        return np.nan

def transform_NRS7(df):
    """Recodes NRS7 to a ordered number scale, 0-10.
    """
    def transform_row(row):
        if row == "Värsta tänkbara värk":
            return 10
        elif row == "Ingen värk":
            return 0
        else:
            return int(row)
    
    df["ini_NRS7"] = df["ini_NRS7"].apply(transform_row)

    return df

def get_column_unique(df):
    col_dict = {}
    for col in df.columns:
        col_dict[col] = df[col].unique().tolist()
    
    return col_dict

