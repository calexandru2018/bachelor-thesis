import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb 

def get_top_10_features(target_params, results, importance_type="weight"):
    """Gets the top 10 features of each XGBoost regressor.
    
    Parameters
    ----------
    target_params: dictionary
        Should contain a dict with with params for each target label.
    results : dictionary
        Should contain a dict with the regression results.
    importance_type: string
        The score type that should be retrieved by. Either weight, gain or cover. Default weight.
    """
    return_dict = {}
    for target in target_params:
        xg_reg = results[target[0]][2]
        ordered_feature = {k: v for k, v in sorted(xg_reg.get_booster().get_score(importance_type=importance_type).items(), key=lambda item: item[1], reverse=True)}
        return_dict[target[0]] = dict(itertools.islice(ordered_feature.items(), 10))
    
    return return_dict, pd.DataFrame.from_dict(return_dict, orient='index')

def sum_feature_appearence(top_10_feat):
    """Sums the ammount of times that a feature appears in a top 10 list of each XGB regressor.
    
    Parameters
    ----------
    top_10_feat : dictionary
        Should contain a dict with with variables for each target label, ordered in desc.
    """
    feature_sum_holder = {}
    for target,feature_dict in top_10_feat.items():
        for feature, feature_score in feature_dict.items():
            if feature in feature_sum_holder:
                feature_sum_holder[feature]["count"] += 1
                feature_sum_holder[feature]["score"] += feature_score
            if not feature in feature_sum_holder:
                feature_sum_holder[feature] = {"count":1, "score": feature_score} 
                
    ordered_dict = {k: v for k, v in sorted(feature_sum_holder.items(), key=lambda item: item[1]["count"], reverse=True)}
    for target, vals in ordered_dict.items():
        ordered_dict[target]["score"] = round((vals["score"] / vals["count"]))
    
    result_dataframe = pd.DataFrame.from_dict(ordered_dict, orient='index')
    result_dataframe.reset_index(level=0, inplace=True)
    result_dataframe.rename({"index":"feature"}, axis=1, inplace=True)
    
    return result_dataframe

def calculate_mean_vif(results):
    """Calculates mean VIF of all target variables.
    
    Parameters
    ----------
    results : Dataframe
        Should contain a VIF dataframe.
    """
    val_holder = {}
    
    for target in results:
        for index, row in results[target][4].iterrows():
            targ_label = row['variables']
            value = row['VIF']
            if not targ_label in val_holder:
                val_holder[targ_label] = {"sum": value, "count": 1}
            else:
                val_holder[targ_label]["sum"] += value
                val_holder[targ_label]["count"] += 1
    
    return_mean_dict = {}
    
    for k,v in val_holder.items():
        return_mean_dict[k] = round((float(v["sum"]) / int(v["count"])),4)
        
    return_desc_mean_dict = {k: v for k, v in sorted(return_mean_dict.items(), key=lambda item: item[1], reverse=True)}

    return pd.DataFrame(return_desc_mean_dict.items(), columns=["Variables", "Mean VIF"])

def generate_vif_barplot(vif_df, save_to_file=False):
    """Generates mean VIF barplot.
    
    Parameters
    ----------
    vif_df : Dataframe
        Should contain all VIF variables, with the mean value for each one.
    save_to_file: Bool
        If True, then .png are saved to local folder.
    """
    plt.figure(figsize=(16,9))
    sns.set_style("whitegrid")

    ax = sns.barplot(x="Variables", y="Mean VIF", data=vif_df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.title('Mean VIF for all target variables')
    plt.show()
    fig = ax.get_figure()
    if save_to_file:
        fig.savefig("mean_vif.png", bbox_inches='tight')

def generate_feature_importance_img(results, save_to_file=False):
    """Generates xgboost features importances images.
    
    Parameters
    ----------
    results : Dataframe
        Should contain the resulting dataframe from run().
    save_to_file: Bool
        If True, then .png are saved to local folder.
    """
        
    plt.rcParams.update({'font.size': 17})
    
    if not os.path.isdir("./feature_importance"):
        os.mkdir("./feature_importance")
        
    for result in results:
        var_name = result

        xg_reg = results[var_name][2]
        
        importance= xg_reg.get_booster().get_score(importance_type='weight')
        
        fig, ax = plt.subplots(figsize=(16,13))
        plot_importance = xgb.plot_importance(importance, importance_type='weight', show_values=True, ax=ax)
        plt.title("Feature importance for "+var_name)
        if save_to_file:
            plot_importance.figure.savefig("./feature_importance/"+var_name+"_feature_importance.png", dpi = 200, bbox_inches='tight')

def generate_tree_img(results, save_to_file=False):
    """Generates xgboost tree images.
    
    Parameters
    ----------
    results : Dataframe
        Should contain the resulting dataframe from run().
    save_to_file: Bool
        If True, then .png are saved to local folder.
    """
    plt.rcParams.update({'font.size': 25})
    
    if not os.path.isdir("./decision_tree"):
        os.mkdir("./decision_tree")
        
    for result in results:
        var_name = result

        xg_reg = results[var_name][2]

        fig, ax = plt.subplots(figsize=(16,10))
        tree = xgb.plot_tree(xg_reg, ax=ax)
        plt.title("Feature importance for "+var_name)
        if save_to_file:
            tree.figure.savefig("./decision_tree/"+var_name+"_tree.png", dpi = 200, bbox_inches='tight')
        
def generate_mean_score_barplot(results, save_to_file=False, return_result_df=False):
    """Generates a barplot with mean feature importance scores.
    
    Parameters
    ----------
    results : Dataframe
        Should contain the resulting dataframe from run().
    save_to_file: Bool
        If True, then .png are saved to local folder.
    return_result_df: Bool
        If True, then the dataframe will be returned.
    """
    feature_importance_sum_dic = {}
    for result in results:
        target_feature = result
        xgb_reg = results[target_feature][2]
        
        importance_dict = xgb_reg.get_booster().get_score(importance_type='weight')
        
        for input_feature, importance_value in importance_dict.items():
            
            if input_feature in feature_importance_sum_dic:
                feature_importance_sum_dic[input_feature]["count"] += 1
                feature_importance_sum_dic[input_feature]["total_importance"] += importance_value
                
            if not input_feature in feature_importance_sum_dic:
                feature_importance_sum_dic[input_feature] = {"count":1, "total_importance": importance_value} 
                
        
    mean_features_importance_dict = {k:round((v["total_importance"])/(v["count"]), 2) for k,v in feature_importance_sum_dic.items()}

    ordered_mean_features_importance_dict = {k:v for k, v in sorted(mean_features_importance_dict.items(), key=lambda item: item[1], reverse=True)}
    
    mean_features_importance_df = pd.DataFrame.from_dict(ordered_mean_features_importance_dict, orient='index')
    
    mean_features_importance_df.reset_index(level=0, inplace=True)
    
    mean_features_importance_df.rename({"index":"input_features", 0:"mean_importance_score"}, axis=1, inplace=True)
        
    plt.figure(figsize=(16,8))
    sns.set_style("whitegrid")

    ax = sns.barplot(x="input_features", y="mean_importance_score", data=mean_features_importance_df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Input Features")
    ax.set_ylabel("Mean Weight")
    plt.title('Mean Feature Importance Score')
    plt.show()
    fig = ax.get_figure()
    if save_to_file:
        fig.savefig("mean_importance_score.png", bbox_inches='tight')
        
    if return_result_df:
        return mean_features_importance_df
