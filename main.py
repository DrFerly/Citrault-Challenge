"""
Axa coding challenge for french third-party motor liability insurance data
ML for Pure Premium
Dr. Lynn Ferres, 15.11.2023
"""
import os
import time
import winsound
import pandas as pd
from tqdm import tqdm
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import analysis_plots
from feature_importance import feature_importances_plot, shap_plots
from global_vars import SHAP, MODEL_PLOTS
from data_preprocessing import encode_string_columns, join_and_clean_data
from actual_vs_pred_plots import actual_pred_feature_plots, actual_pred_scatter
from model_metrics import calculate_model_metrics, \
    confusion_matrix_and_class_report, score_estimator


def main():
    """"
    main function to load, join and clean the data, train models, make predictions
    and create all kind of plots necessary for data analysis and model evaluation.
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
    # create plots for data analysis if CREATE_ANALYSIS_PLOTS is True
    analysis_plots.analysis_plots(df, plotpath, string_columns)
    # drop features that should not be considered in model
    df = df.drop(columns=["Area"])
    # try out different feature sets and loop over selection:
    selected_features_to_drop = [
        ["IDpol", "VehGas", "VehBrand", "BonusMalus", "Frequency", "Exposure", "VehPower", "VehAge"],
        # ["IDpol", "VehGas", "Density", "DrivAge"],
        # ["IDpol"],
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
        all_targets = [
                       'ClaimNb',
                       'ClaimAmount',
                       # 'Frequency',
                       # 'ClaimAmount'
                       ]
        # all_targets = ['ClaimAmount']
        # GLM with Poisson ditribution and log link
        # reg = TweedieRegressor(power=1, alpha=0.5, link='log')
        all_algos = [
                     RandomForestClassifier(n_estimators=2500),
                     RandomForestRegressor(),
                     # PoissonRegressor(alpha=1e-4, solver="newton-cholesky"),
                     # GammaRegressor(alpha=10.0, solver="newton-cholesky")
                     ]
        # all_algos =[GammaRegressor(alpha=10.0, solver="newton-cholesky")]
        # set start of progress bar
        pbar = tqdm(total=100)
        for target, alg in zip(all_targets, all_algos):
            if 'Gamma' in str(alg):
                # gamma needs positive values
                df = df[df["ClaimAmount"]>0]
                df["ClaimAmount"] = df["ClaimAmount"] / df["ClaimNb"]
            y = df[target]
            X = df.drop(all_targets, axis=1)
            if 'Poisson' not in str(alg) and "Frequency" in X.columns:
                X = X.drop(columns=["Frequency"])
            # Instantiate scaler and fit on features
            scaler = StandardScaler()
            scaler.fit(X)
            X_scaled = scaler.transform(X)
            col_names = X.columns
            X_scaled = pd.DataFrame(data=X_scaled, columns=col_names)
            # split into train and test
            model = alg
            print("fitting model")
            if 'Gamma' in str(alg):
                X_train, X_test, y_train, y_test = train_test_split(
                    df,
                    X_scaled,
                    test_size=0.2,
                    random_state=0)
                model.fit(
                    y_train, X_train["ClaimAmount"],
                    sample_weight=X_train["ClaimNb"]
                    )
                scores = score_estimator(
                    model,
                    y_train,
                    y_test,
                    X_train,
                    X_test,
                    target="ClaimAmount",
                    weights="ClaimNb",
                )
                print("Evaluation of GammaRegressor on target AvgClaimAmount")
                print(scores)
                if MODEL_PLOTS:
                    actual_pred_feature_plots(
                        model,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        target,
                        plotpath,
                        "ClaimNb"
                    )
                data_mean = df_initial[target].mean()
                mean_claims_greater_one = X_train[target][X_train[target] > 0].mean()
                model_mean = model.predict(y_train).mean()
                print("Mean ClaimAmount: ",  data_mean)
                print("Mean ClaimAmount for Claims: ",  mean_claims_greater_one)
                print("pedicted mean ClaimAmount for Claims: ",  model_mean)
            elif 'Poisson' in str(alg):
                X_train, X_test, y_train, y_test = train_test_split(
                    df,
                    X_scaled,
                    test_size=0.2,
                    random_state=0
                )
                model.fit(y_train, X_train["Frequency"], sample_weight=X_train["Exposure"])
                scores = score_estimator(
                    model,
                    y_train,
                    y_test,
                    X_train,
                    X_test,
                    target="Frequency",
                    weights="Exposure",
                )
                print("Evaluation of PoissonRegressor on target Frequency")
                print(scores)
                if MODEL_PLOTS:
                    actual_pred_feature_plots(
                        model,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        target,
                        plotpath,
                        "Exposure"
                    )
                data_mean = df_initial["ClaimNb"].mean()
                mean_for_claims_greater_than_one = X_train["ClaimNb"][X_train["ClaimNb"] > 0].mean()
                model_mean = model.predict(y_train).mean()
                print("Mean NbClaims: ",  data_mean)
                print("Mean NbClaims for ClaimAmount>0: ",  mean_for_claims_greater_than_one)
                print("absoluteNbOfClaims", len(X_train[target][X_train[target] > 0]))
                print("pedicted mean Frequency: ",  model_mean)
            else:
                # if RandomForest:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
                mod = model.fit(X_train, y_train)
                # svm = SVC(kernel='linear')
                # glm_mod = reg.fit(X_train, y_train)
                # svm.fit(X_train, y_train)
                feature_importances = pd.Series(model.feature_importances_, \
                    index=X.columns).sort_values(ascending=False)
                # shap values TODO store them since it takes forever to compute!
                if SHAP:
                    print("calculating Shap values")
                    # X_test_backscaled = scaler.inverse_transform(X_test)
                    # only take 100 otherwise it takes too long
                    X100 = shap.utils.sample(X_test, 100)
                    explainer = shap.Explainer(model.predict, X100)
                    shap_values = explainer(X_test)
                # shap_values = explainer.shap_values(X_test)
                print("prediction in progress")
                y_pred = model.predict(X_test)
                # glm_y_pred = reg.coef_
                # svm_pred = svm.predict(X_test)
                y_test = pd.DataFrame(y_test)
                y_pred = pd.DataFrame(y_pred, columns=[target])
                print("create plot actual vs predicted")
                # model plots and metrics
                if MODEL_PLOTS:
                    actual_pred_scatter(y_test, y_pred, target, plotpath, alg)
                    feature_importances_plot(feature_importances, plotpath, alg)
                actual_pred_feature_plots(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    target,
                    plotpath,
                    None
                )
                confusion_matrix_and_class_report(
                    alg,
                    y_test,
                    y_pred,
                    plotpath,
                    feat_sel,
                    target
                )
                if SHAP:
                    shap_plots(alg, y_pred, X_test, shap_values, plotpath)
                # confusion matrix and classification report
                print("calculate accuracy score")
                calculate_model_metrics(
                    mod,
                    X_train,
                    y_train,
                    target,
                    X_test,
                    y_test
                    )
            # update progress bar for iteration through models
            pbar.update(100/nb_of_iterations)
        pbar.close()
    # tell me when it's done:
    winsound.Beep(frequency=440, duration=1000)
    # get and display runtime
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time/60} minutes")
    print('done!')


if __name__ == "__main__":
    main()
