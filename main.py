"""
coding challenge for french car insurance
"""
import pandas as pd
import os
from counts import counts
from correlations import plot_correlations, scatter
from boxplots import boxplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

CREATE_PLOTS = False

def main():
    """"
    main function to load data
    """
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
    # df = df_claims.merge(df_damage, on=["IDpol"], how='left')
    # df = df[~df["ClaimAmount"].isnull()]
    # create plots for data analysis
    plots(df, plotpath, string_columns)
    # data reformatting for model training
    df = df_claims
    columns_to_encode = ['Region', 'VehBrand', 'VehGas', 'Area']
    for col_to_enc in columns_to_encode:
        encode_string_columns(col_to_enc, df)
    # count nan's and tell me where they are:
    if df.isna().sum().sum()>0:
        for col in df.columns:
            print(str(df[col].isna().sum()) + " NaN's in col " + str(col))
    # df.drop(columns="ClaimAmount")
    # define train and test set for model training
    target = 'ClaimNb'
    y = df[target]
    X = df.drop(target, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rfc = RandomForestClassifier()
    print("fitting model")
    rfc.fit(X_train, y_train)
    print("prediction in progress")
    y_pred = rfc.predict(X_test)
    print("calculate accuracy score")
    accuracy = accuracy_score(y_test, y_pred)
    print("The Accuracy for predicting " + target + " is: ", accuracy)
    print(df)


def encode_string_columns(colname, df):
    """
    doc str
    """
    region_array = df[colname].unique()
    int_array = []
    dict_region_mapping={}
    for i in range(0, len(region_array)):
        int_array.append(i)
    for region, i in zip(region_array, int_array):
        dict_region_mapping.update({region:i})
    df[colname] = df[colname].map(dict_region_mapping)


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
