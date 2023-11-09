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
# from sklearn.neighbors import NearestNeighbors as knn
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from analysis_plots import boxplot, plot_correlations, scatter, counts, confusion_matrix_plot
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import winsound
from tqdm import tqdm
import shap

CREATE_PLOTS = False

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
    ###### TODO reactivate the following lines later ###############
    print(df_claims['ClaimNb'].value_counts())
    df_claims = df_claims[df_claims["ClaimNb"]<=3]
    df = df_claims.merge(df_damage, on=["IDpol"], how='left')
    # df = df[~df["ClaimAmount"].isnull()]
    # create plots for data analysis
    plots(df, plotpath, string_columns)
    # data reformatting for model training
    df_initial = df_claims
    selected_features_to_drop = [
        ["IDpol"],
        # ["IDpol", "VehGas", "Density", "DrivAge"],
        # ["IDpol", "Area", "Region"],
        # ["IDpol", "VehBrand", "Area", "Region"]
    ]

    pbar = tqdm(total=100)
    nb_of_iterations = len(selected_features_to_drop)
    for feat_sel in selected_features_to_drop:
        df = df_initial.drop(columns=feat_sel)
        columns_to_encode = [
            'Region',
            'VehBrand',
            'VehGas',
            'Area'
            ]
        for col_to_enc in columns_to_encode:
            if not col_to_enc in feat_sel:
                df = encode_string_columns(col_to_enc, df)
        # count nan's and tell me where they are:
        if df.isna().sum().sum()>0:
            for col in df.columns:
                print(str(df[col].isna().sum()) + " NaN's in col " + str(col))
        # df.drop(columns="ClaimAmount")
        # define train and test set for model training
        # df = df.drop(columns=["IDpol", "Region", "Vehmodel"])
        target = 'ClaimNb'
        y = df[target]
        X = df.drop(target, axis=1)
        # Instantiate scaler and fit on features
        scaler = StandardScaler()
        scaler.fit(X)

        # Transform features
        X_scaled = scaler.transform(X)


        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
        rfc = RandomForestClassifier()
        # svm = SVC(kernel='linear')
        print("fitting model")
        rfc.fit(X_train, y_train)
        print("fitting svm model")
        # svm.fit(X_train, y_train)
        feature_importances = pd.Series(rfc.feature_importances_, index=X.columns).sort_values(ascending=False)

        print("prediction in progress")
        y_pred = rfc.predict(X_test)
        # svm_pred = svm.predict(X_test)
        # X_test["y_pred"] = list(y_pred)
        # X_test["y_test"] = list(y_test)
        print("create plot actual vs predicted")
        model_plotpath = os.path.join(plotpath, "models")
        modelname="RandomForestClassifier"
        scatter_model_filename = modelname + str(feat_sel) + '.png'
        outfile = os.path.join(model_plotpath, scatter_model_filename)
        scatter(
            list(y_test),
            list(y_pred),
            xlabel="actual",
            ylabel="prediction",
            outfile=outfile,
            close=False
            )
        if 'Class' in modelname:
            confusion_matrix_plot(y_test, y_pred, plotpath, str(feat_sel))
        print("calculate accuracy score")
        scores = rfc.score(X, y, sample_weight=None)
        accuracy = accuracy_score(y_test, y_pred)
        # precision = precision_score(y_test, y_pred)
        # recall = recall_score(y_test, y_pred)
        print("The Accuracy for predicting " + target + " is: ", accuracy)
        feature_importances.plot.bar
        model_preds = {
            # "Logistic Regression": log_reg_preds,
            # "Support Vector Machine": svm_pred,
            "Random Forest Classifier": y_pred
            }
        for model, preds in model_preds.items():
            print(f"{model} Results:\n{classification_report(y_test, preds)}", sep="\n\n")
        pbar.update(100/nb_of_iterations)
    pbar.close()
    # tell me when you're done:
    classification_report(y_test, y_pred)
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time/60} minutes")

    print('done!')


def encode_string_columns(colname, df):
    """
    doc str
    """
    df_encoded = pd.get_dummies(df[colname])
    df = df.drop(columns=colname)
    df = df.join(df_encoded)
    return df
    # region_array = df[colname].unique()
    # int_array = []
    # dict_region_mapping={}
    # for i in range(0, len(region_array)):
    #     int_array.append(i)
    # for region, i in zip(region_array, int_array):
    #     dict_region_mapping.update({region:i})
    # df[colname] = df[colname].map(dict_region_mapping)


def plots(df, plotpath, string_columns):
    """
    create plots for data analysis
    """
    if CREATE_PLOTS:
        boxplot(df, plotpath, string_columns)
        counts(df, plotpath)
        plot_correlations(df, plotpath, plotpath, string_columns)
        targets = ["ClaimNb", "ClaimAmount"]
        scatterplotpath = os.path.join(plotpath, "scatter")
        for tar in targets:
            df_tar = df
            if tar == "ClaimNb":
                df_tar = df.drop(columns=["ClaimAmount"]).drop_duplicates()
            for tcol in df_tar.columns:
                scatter_filename = tar + "_and_" + tcol + '.png'
                outfile = os.path.join(scatterplotpath, scatter_filename)
                scatter(
                    df[tcol],
                    df[tar],
                    xlabel=tcol,
                    ylabel=tar,
                    outfile=outfile,
                    close=True
                )


if __name__ == "__main__":
    main()

