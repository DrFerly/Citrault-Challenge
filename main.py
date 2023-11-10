"""
coding challenge for french car insurance
"""
import os
from scipy.stats import randint
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from analysis_plots import analysis_plots, scatter, confusion_matrix_plot, actual_pred_scatter
from GLOBAL_VAR import CREATE_PLOTS, SHAP
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import winsound
from tqdm import tqdm
import shap



def main():
    """"
    main function to load data
    """
    shap.initjs()
    start_time = time.perf_counter()
    # paths
    base_path = r"C:\Users\LILEL\Desktop\Citrault-Challenge"
    claims_filepath = r"C:\Users\LILEL\Desktop\Citrault-Challenge\raw_data\freMTPL2freq.csv"
    plotpath = os.path.join(base_path, "plots")
    damage_filepath = r"C:\Users\LILEL\Desktop\Citrault-Challenge\raw_data\freMTPL2sev.csv"
    # read data
    df_claims = pd.read_csv(claims_filepath)
    df_damage = pd.read_csv(damage_filepath)
    # often occuring entries
    # df_damage = df_damage[df_damage["ClaimAmount"] != 1204]
    # df_damage = df_damage[df_damage["ClaimAmount"] != 1128]
    # df_damage = df_damage[df_damage["ClaimAmount"] != 602]
    # df_damage = df_damage[df_damage["ClaimAmount"] != 2408]
    # clean data and format columns of string type:
    df_claims = df_claims.replace('\'', '')
    string_columns = ['Area', 'VehBrand', 'VehGas', 'Region']
    for str_col in string_columns:
        df_claims[str_col] = df_claims[str_col].apply(lambda x: x.replace("'", ""))
    df_claims["IDpol"] = df_claims["IDpol"].astype(int)
    df_claims["IDpol"] = df_claims["IDpol"].astype('string')
    df_damage["IDpol"] = df_damage["IDpol"].astype(str)
    print(df_claims['ClaimNb'].value_counts())
    df_claims = df_claims[df_claims["ClaimNb"]<=3]
    df_claims = df_claims[df_claims["Exposure"]<=1]
    df_claims["Density"] = np.log(df_claims["Density"])
    df = df_claims.merge(df_damage, on=["IDpol"], how='left')
    df_total_amount = df[["IDpol", "ClaimAmount"]].groupby(['IDpol']).sum()
    df = df_claims.merge(df_total_amount, on=["IDpol"], how='left')
    df["ClaimAmount"] = df["ClaimAmount"].replace(np.nan, 0)
    # df["Total_ClaimAmount"] = df["ClaimNb"] * df["ClaimAmount"]
    # df = df[~df["ClaimAmount"].isnull()]
    # create plots for data analysis
    analysis_plots(df, plotpath, string_columns)
    # data reformatting for model training
    df = df.drop(columns=["IDpol", "Area", "Exposure"])
    df_initial = df
    pbar = tqdm(total=100)
    selected_features_to_drop = [
        ["BonusMalus"],
        # ["IDpol", "VehGas", "Density", "DrivAge"],
        # ["IDpol", "Area", "Region"],
        # ["IDpol", "VehBrand", "Area", "Region"]
    ]
    nb_of_iterations = len(selected_features_to_drop)
    for feat_sel in selected_features_to_drop:
        df = df_initial.drop(columns=feat_sel)
        columns_to_encode = [
            'Region',
            'VehBrand',
            'VehGas',
            # 'Area'
            ]
        for col_to_enc in columns_to_encode:
            if not col_to_enc in feat_sel:
                df = encode_string_columns(col_to_enc, df)
        # count nan's and tell me where they are:
        if df.isna().sum().sum()>0:
            for col in df.columns:
                print(str(df[col].isna().sum()) + " NaN's in col " + str(col))
        # define targets and algorithms to try
        all_targets = ['ClaimNb', 'ClaimAmount']
        all_algos = [RandomForestClassifier(), RandomForestRegressor()]
        # for target, alg in zip(all_target, all_alogs):
        target = 'ClaimAmount'
        alg=RandomForestRegressor()
        y = df[target]
        X = df.drop(all_targets, axis=1)
        # Instantiate scaler and fit on features
        scaler = StandardScaler()
        scaler.fit(X)
        # X_scaled = scaler.transform(X)
        col_names = X.columns
        # X_scaled = pd.DataFrame(data=X_scaled, columns=col_names)
        # split into train and test
        # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.002)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        rfc = alg
        # svm = SVC(kernel='linear')
        print(" ")
        print("fitting model")
        mod = rfc.fit(X_train, y_train)
        print("calculating Shap values")
        # svm.fit(X_train, y_train)
        feature_importances = pd.Series(rfc.feature_importances_, index=X.columns).sort_values(ascending=False)
        # shap values
        if SHAP:
            X100 = shap.utils.sample(X_test, 100)
            explainer = shap.Explainer(rfc.predict, X100)
            shap_values = explainer(X_test)
        # shap_values = explainer.shap_values(X_test)
        print("prediction in progress")
        y_pred = rfc.predict(X_test)
        # svm_pred = svm.predict(X_test)
        # X_test["y_pred"] = list(y_pred)
        # X_test["y_test"] = list(y_test)
        y_test = pd.DataFrame(y_test)
        y_pred = pd.DataFrame(y_pred, columns=[target])
        print("create plot actual vs predicted")
        model_plotpath = os.path.join(plotpath, "models")
        modelname="RandomForestClassifier"
        scatter_model_filename = modelname + str(target) + '.png'
        outfile = os.path.join(model_plotpath, scatter_model_filename)
        actual_pred_scatter(y_test, y_pred, target, outfile)
        if 'Class' in str(alg):
            confusion_matrix_plot(y_test, y_pred, plotpath, str(feat_sel))
            accuracy = accuracy_score(y_test, y_pred)
            print("The Accuracy for predicting " + target + " is: ", accuracy)
            model_preds = {
            # "Logistic Regression": log_reg_preds,
            # "Support Vector Machine": svm_pred,
            "Random Forest Classifier": y_pred
            }
            for model, preds in model_preds.items():
                print(f"{model} Results:\n{classification_report(y_test, preds)}", sep="\n\n")
            
        print("calculate accuracy score")
        train_scores = mod.score(X_train, y_train, sample_weight=None)
        print("The Accuracy for training " + target + " is: ", train_scores)
        test_scores = mod.score(X_test, y_test, sample_weight=None)
        print("The Accuracy for predicting " + target + " is: ", test_scores)
        # precision = precision_score(y_test, y_pred)
        # recall = recall_score(y_test, y_pred)
        plt.figure(figsize=(20, 20))
        feature_importances.plot.bar()
        model_plotpath = os.path.join(plotpath, "models")
        feat_file_name="Feature_importances"
        features_model_filename = modelname + feat_file_name + '.png'
        featfile = os.path.join(model_plotpath, features_model_filename)
        plt.savefig(featfile)
        plt.close("all")
        pbar.update(100/nb_of_iterations)
    if SHAP:
        if 'Class' in str(alg):
            example_for_1 = np.where(y_pred == 1)[0][0]
            example_for_0 = np.where(y_pred == 1)[0][0]
            shap.plots.waterfall(shap_values[example_for_1], max_display=20)
            shap.plots.waterfall(shap_values[example_for_0], max_display=20)
        shap.summary_plot(shap_values, X_test)
        shap.plots.beeswarm(shap_values)
        shap.plots.beeswarm(shap_values.abs, color="shap_red")
        shap.plots.bar(shap_values)
        shap.plots.bar(shap_values.abs.max(0))
        shap.plots.scatter(shap_values[:, "DrivAge"], color=shap_values[:, "VehAge"])
        for col in X_test.columns:
            shap.plots.scatter(shap_values[:, "Density"])
    pbar.close()
    # tell me when it's done:
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
    # get runtime
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time/60} minutes")

    print('done!')


def encode_string_columns(colname, df):
    """
    doc str
    """
    # df_encoded = pd.get_dummies(df[colname])
    # df = df.drop(columns=colname)
    # df = df.join(df_encoded)
    # return df
    region_array = df[colname].unique()
    int_array = []
    dict_region_mapping={}
    for i in range(0, len(region_array)):
        int_array.append(i)
    for region, i in zip(region_array, int_array):
        dict_region_mapping.update({region:i})
    df[colname] = df[colname].map(dict_region_mapping)
    return df



if __name__ == "__main__":
    main()

