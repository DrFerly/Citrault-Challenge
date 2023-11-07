"""
coding challenge for french car insurance
"""
import pandas as pd
import os
from counts import counts
from correlations import plot_correlations, scatter
from boxplots import boxplot
from sklearn.ensemble import RandomForestClassifier

CREATE_PLOTS = True

def main():
    """"
    main function to load data
    """
    base_path = r"C:\Users\LILEL\Desktop\Citrault-Challenge"
    claims_filepath = r"C:\Users\LILEL\Desktop\Citrault-Challenge\raw_data\freMTPL2freq.csv"
    plotpath = os.path.join(base_path, "plots")
    damage_filepath = r"C:\Users\LILEL\Desktop\Citrault-Challenge\raw_data\freMTPL2sev.csv"
    df_claims = pd.read_csv(claims_filepath)
    df_damage = pd.read_csv(damage_filepath)
    # often occuring entries
    # df_damage = df_damage[df_damage["ClaimAmount"] != 1204]
    # df_damage = df_damage[df_damage["ClaimAmount"] != 1128]
    # df_damage = df_damage[df_damage["ClaimAmount"] != 602]
    # df_damage = df_damage[df_damage["ClaimAmount"] != 2408]
    df_claims = df_claims.replace('\'', '')
    string_columns = ['Area', 'VehBrand', 'VehGas', 'Region']
    for str_col in string_columns:
        df_claims[str_col] = df_claims[str_col].apply(lambda x: x.replace("'", ""))
    df = df_claims.merge(df_damage, on=["IDpol"], how='left')
    plots(df, plotpath, string_columns)
    print(df)


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
