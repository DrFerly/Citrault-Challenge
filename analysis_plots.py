"""
script to produce histogram plots of feature columns
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def scatter(x, y, xlabel=None, ylabel=None, outfile=None, close=True):
    """
    function to create a scatterplot using the scatter function
    """
    plt.plot(x, y, marker="o", linewidth = 0, markersize=10)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.tick_params(axis='both', which='minor', labelsize=40)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=40)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=40)
    if outfile:
        plt.savefig(outfile)
    if close:
        plt.close("all")


def confusion_matrix_plot(y_test, y_pred, outpath, feat_sel):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    cmpath = os.path.join(outpath, "confusion_matrix")
    outname = feat_sel + "cm.png"
    plt.savefig(os.path.join(cmpath, outname))
    plt.close("all")


def heatmap(data_joined, correlations, heatmap_path, plt_nb):
    """"
    creates a heatmap plot for all the properties

    :param data_joined: input data for correlation calculation
    :type data_joined: pandas df of ints
    :param correlations: correlation df from df.corr()
    :type data_joined: pandas df of floats
    :param heatmap_path: output path for heatmap png
    :type heatmap_path: string
    """
    # create plot
    subplotsize = [30., 30.]
    figuresize = [40., 40.]
    fig = plt.figure(figsize=figuresize)
    ax = fig.add_subplot(3, 1, (1, 2), yticklabels=(-1, 1, 0.5))
    left = 0.5*(1.-subplotsize[0]/figuresize[0])
    right = 1.-left
    bottom = 0.5*(1.-subplotsize[1]/figuresize[1])
    top = 1.-bottom
    fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.tick_params(axis='both', which='minor', labelsize=40)
    # plot the correlations
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    # plot the colorbar
    cb1 = plt.colorbar(cax)
    cb1.ax.tick_params(labelsize =40)
    lengthx = len(data_joined.columns)
    lengthy = len(data_joined.index)
    plt.xticks(np.arange(0, lengthx, 1), data_joined.columns, rotation='vertical')
    plt.yticks(np.arange(0, lengthy, 1), data_joined.index)
    # save plot as png
    heatmap_path = os.path.join(heatmap_path, "heatmap")
    outname = str(plt_nb) + r"heatmap.png"
    plt.savefig(os.path.join(heatmap_path, outname))
    plt.close("all")


def plot_correlations(df_ingreds_np, plotpath, heatmap_path, string_columns):
    """
    creates correlations, a heatmap plot and scatter plots for each column pair correlations
    
    :param df_ingred_np: numpy df of the ingredients
    :type df_ingred_np: pd.DataFrame
    :param plotpath: output path for the plot folder
    :type plotpath: string
    :param heatmap_path: output path for the heatmap.png
    :type heatmap_path: string
    """
    # create df, rename cols, drop not needed cols
    df = pd.DataFrame(df_ingreds_np, columns=df_ingreds_np.columns)
    counts_renaming = {}
    for col in df.columns:
        special_chars = ['\n', '.', ',', ':', '/', '\'']
        for spec_char in special_chars:
            col_cleaned = col.replace(spec_char, '')
            counts_renaming.update({col : col_cleaned})
    df = df.rename(columns=counts_renaming)
    df.replace(0, np.nan, inplace=True)
    df.replace('/', '', inplace=True)

    df = df.drop(string_columns, axis=1)
    column_selection = df.columns
    # select dataset for correlations
    df_corr = df[column_selection]
    df_corr = df_corr.apply(pd.to_numeric)
    # calculate correltion
    correlations = df_corr.corr()
    correlations = correlations.dropna(axis=1, how='all')
    correlations = correlations.dropna(axis=0, how='all')

    # # PLA specific code (following 4 lines of code)
    # ingredient_correlations = correlations.loc[:'Epon 164', :'Epon 164']
    # property_correlations = correlations.loc['RPM':, 'RPM':]
    # ing_prop_correlations = correlations.loc['RPM':, :'Epon 164']
    # # import ipdb; ipdb.set_trace()
    plt_nb = 0
    # for correl in [ingredient_correlations, property_correlations, ing_prop_correlations]:
    #     plt_nb += 1
    #     # create the heatmap plot
    #     heatmap(correl, correl, heatmap_path, plt_nb)
    #     # create the scatter plots:
    heatmap(correlations, correlations, heatmap_path, plt_nb)
    columns = correlations.columns
    for k in range(0, len(correlations)):
        for l in range(k+1, len(correlations)):
            col_n1 = columns[k]
            col_n2 = columns[l]
            # print all correlations
            print("Observed correlation of %.2f for %s and %s" % (
                correlations[col_n1][col_n2],
                col_n1,
                col_n2
                ))
            # select to save ALL correlations, or ONLY high correlations:
            # outname = r"scatter plots\high correlation\correlation_%s_%s.png" % (col_n1, col_n2)
            outname = r"correlation_%s_%s.png" % (col_n1, col_n2)
            corr_plotpath = os.path.join(plotpath, "correlation")
            outfile = os.path.join(corr_plotpath, outname)
            if (correlations[col_n1][col_n2] > 0.75) | (correlations[col_n1][col_n2] < -0.75):
                scatter(
                    df_corr[col_n1],
                    df_corr[col_n2],
                    xlabel=col_n1,
                    ylabel=col_n2,
                    outfile=outfile,
                    close=True
                )
            plt.close("all")


def boxplot(df, plotpath, string_columns):
    """"
    creates a boxplot each column

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
    df = df.drop(columns=string_columns)
    boxplotpath = os.path.join(plotpath, "boxplot")
    # for each columns do the counts:
    for prop in list(df.columns):
        # summe is the sum of the counts of each column ingredient
        # summe = df[prop].count()
        # elem is the column elements
        elem = list(df[prop][np.invert(pd.isnull(df[prop]))].to_numpy())
        if prop == "ClaimAmount":
            # elem = np.log10(elem)
            prop = "log(ClaimAmount)"
        # create the plot
        plt.figure(figsize=(20, 20))
        plt.boxplot(elem)
        plt.title("boxplot of " + str(prop), fontsize=40)
        # create the histogram
        # plt.hist(elem, bins=10, histtype="stepfilled", color="gray")
        ax = plt.gca()
        if 'log' in prop:
            plt.yscale('log')
        # ax.tick_params(axis='both', which='major', labelsize=40)
        # ax.tick_params(axis='both', which='minor', labelsize=40)
        ax.set_ylabel(str(prop), fontsize=40)
        # ax.set_xlabel(prop, fontsize=40)
        # plt.xticks(rotation='vertical')
        # save plot
        outname = "boxplot of " + str(prop) + ".png"
        plt.savefig(os.path.join(boxplotpath, outname))
        plt.close("all")


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
    countsplotpath = os.path.join(countsplotpath, "histogram")
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
