import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def counts(df, countsplotpath):
    """"
    creates a histogram plot of the counts for each ingredient column

    :param df: input dataframe with the ingredients as columns
    :type df: pd.DataFrame
    :param countsplotpath: output path for plots
    :type countsplotpath: string
    """
    # use the renaming dictionary in order to get rid of the special characters#
    counts_renaming = {}
    for col in df.columns:
        special_chars = ['\n', '.', ',', ':', '/', '\'']
        for spec_char in special_chars:
            col_cleaned = col.replace(spec_char, '')
            counts_renaming.update({col : col_cleaned})
    df = df.rename(columns=counts_renaming)
    df.replace(0, np.nan, inplace=True)
    # for each columns do the counts:
    for prop in list(df.columns):
        # summe is the sum of the counts of each column ingredient
        summe = df[prop].count()
        # elem is the column elements
        elem = list(df[prop][np.invert(pd.isnull(df[prop]))].to_numpy())
        # create the plot
        plt.figure(figsize=(20, 20))
        plt.title("sum for " + str(prop) + " : " + str(summe), fontsize=40)
        # create the histogram
        plt.hist(elem, bins=10, histtype="stepfilled", color="gray")
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=40)
        ax.tick_params(axis='both', which='minor', labelsize=40)
        ax.set_ylabel("count", fontsize=40)
        ax.set_xlabel(prop, fontsize=40)
        plt.xticks(rotation='vertical')
        # save plot
        outname = str(summe) + "x " + str(prop) + ".png"
        plt.savefig(os.path.join(countsplotpath, outname))
        plt.close("all")
