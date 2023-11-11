"""
coding challenge for french car insurance
"""
import os
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
import winsound
from tqdm import tqdm
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from analysis_plots import analysis_plots, actual_pred_scatter, shap_plots, feature_importances_plot, confusion_matrix_and_class_report, calculate_model_metrics
from GLOBAL_VAR import SHAP, MODEL_PLOTS
from sklearn.preprocessing import StandardScaler
from data_preprocessing import encode_string_columns, join_and_clean_data



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
    string_columns = ['Area', 'VehBrand', 'VehGas', 'Region']
    # clean data and format columns of string type:
    df = join_and_clean_data(df_claims, df_damage, string_columns)
    # create plots for data analysis
    analysis_plots(df, plotpath, string_columns)
    # drop features that should not be considered in model
    df = df.drop(columns=["IDpol", "Area", "Exposure"])
    # set start if progress bar
    pbar = tqdm(total=100)
    # try out different features and loop over selection for model training+ pred:
    selected_features_to_drop = [
        ["BonusMalus"],
        # ["IDpol", "VehGas", "Density", "DrivAge"],
        # ["IDpol", "Area", "Region"],
        # ["IDpol", "VehBrand", "Area", "Region"]
    ]
    nb_of_iterations = len(selected_features_to_drop)
    df_initial = df
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
        for target, alg in zip(all_targets, all_algos):
            # target = 'ClaimAmount'
            # alg = RandomForestRegressor()
            y = df[target]
            X = df.drop(all_targets, axis=1)
            # Instantiate scaler and fit on features
            # scaler = StandardScaler()
            # scaler.fit(X)
            # X_scaled = scaler.transform(X)
            # col_names = X.columns
            # X_scaled = pd.DataFrame(data=X_scaled, columns=col_names)
            # split into train and test
            # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.002)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.002)
            rfc = alg
            # svm = SVC(kernel='linear')
            print("fitting model")
            mod = rfc.fit(X_train, y_train)
            # svm.fit(X_train, y_train)
            feature_importances = pd.Series(rfc.feature_importances_, index=X.columns).sort_values(ascending=False)
            # shap values TODO store them since it takes forever to compute!
            if SHAP:
                print("calculating Shap values")
                # X_test_backscaled = scaler.inverse_transform(X_test)
                X100 = shap.utils.sample(X_test, 100)
                explainer = shap.Explainer(rfc.predict, X100)
                shap_values = explainer(X_test)
            # shap_values = explainer.shap_values(X_test)
            print("prediction in progress")
            y_pred = rfc.predict(X_test)
            # svm_pred = svm.predict(X_test)
            y_test = pd.DataFrame(y_test)
            y_pred = pd.DataFrame(y_pred, columns=[target])
            print("create plot actual vs predicted")
            # model plots and metrics
            if MODEL_PLOTS:
                actual_pred_scatter(y_test, y_pred, target, plotpath, alg)
                feature_importances_plot(feature_importances, plotpath, alg)
                confusion_matrix_and_class_report(alg, y_test, y_pred, plotpath, feat_sel, target)
            if SHAP:
                shap_plots(alg, y_pred, X_test, shap_values, plotpath)
            # confusion matrix and classification report
            print("calculate accuracy score")
            calculate_model_metrics(mod, X_train, y_train, target, X_test, y_test, alg)
            pbar.update(100/nb_of_iterations)
    # update progress bar for iteration through models
    pbar.close()
    # tell me when it's done:
    winsound.Beep(frequency=440, duration=1000)
    # get runtime
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time/60} minutes")

    print('done!')





if __name__ == "__main__":
    main()
