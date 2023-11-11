"""
script to produce histogram plots of feature columns
"""
import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from GLOBAL_VAR import CREATE_ANALYSIS_PLOTS, SHAP
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


def analysis_plots(df, plotpath, string_columns):
    """
    create plots for data analysis
    """
    if CREATE_ANALYSIS_PLOTS:
        print("create boxplots")
        boxplot(df, plotpath, string_columns)
        print("create count plots")
        counts(df, plotpath)
        print("create correlation plots")
        plot_correlations(df, plotpath, plotpath, string_columns)
        targets = ["ClaimNb", "ClaimAmount"]
        scatterplotpath = os.path.join(plotpath, "scatter")
        for tar in targets:
            df_tar = df
            df_tar = df_tar.drop(columns=["IDpol"])
            for tcol in df_tar.columns:
                if tcol != tar:
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


def actual_pred_scatter(y_test, y_pred, target, plotpath, alg):
    model_plotpath = os.path.join(plotpath, "models")
    modelname=str(alg)
    scatter_model_filename = modelname + str(target) + '.png'
    outfile = os.path.join(model_plotpath, scatter_model_filename)
    fig = plt.figure(figsize=(25, 25))
    ax1 = fig.add_subplot(111)
    ax1.scatter(y_test[target], y_pred[target], s=20, c='b', marker="o", label='RFRegressor')
    # ax1.scatter(x_id, y_pred[target], s=10, c='r', marker="o", label='Predicted')
    plt.legend(loc="upper right", fontsize=40)
    ax1.set_ylabel("predicted", fontsize=40)
    ax1.set_xlabel("actual", fontsize=40)
    ax1.tick_params(axis='both', which='major', labelsize=40)
    ax1.tick_params(axis='both', which='minor', labelsize=40)
    ax1.axline((0, 0), slope=0.5, c='black')
    ax1.set_xlim(0, 5000)
    ax1.set_ylim(0, 5000)
    # ax1.margins(50)
    if outfile:
        plt.savefig(outfile)
    plt.close("all")



def scatter(x, y, xlabel=None, ylabel=None, outfile=None, close=True):
    """
    function to create a scatterplot using the scatter function
    """
    plt.figure(figsize=(20, 20))
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
    plt_nb = 0
    # for correl in [ingredient_correlations, property_correlations, ing_prop_correlations]:
    #     plt_nb += 1
    #     # create the heatmap plot
    #     heatmap(correl, correl, heatmap_path, plt_nb)
    #     # create the scatter plots:
    heatmap(correlations, correlations, heatmap_path, plt_nb)
    plt.close("all")
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
            # if (correlations[col_n1][col_n2] > 0.3) | (correlations[col_n1][col_n2] < -0.3):
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
    for prop in list(df.columns.drop("IDpol")):
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
        ax.tick_params(axis='both', which='major', labelsize=40)
        ax.tick_params(axis='both', which='minor', labelsize=40)
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
    # counts_renaming = {}
    # for col in df.columns:
    #     special_chars = ['\n', '.', ',', ':', '/', '\'']
    #     for spec_char in special_chars:
    #         col_cleaned = col.replace(spec_char, '')
    #         counts_renaming.update({col : col_cleaned})
    # df = df.rename(columns=counts_renaming)
    # df.replace(0, np.nan, inplace=True)
    countsplotpath = os.path.join(countsplotpath, "histogram")
    # for each columns do the counts:
    for prop in list(df.columns.drop("IDpol")):
        if prop == "ClaimAmount":
            nb_bins= 30
            df_copy = df[df["ClaimAmount"]>0]
            df_copy = df_copy[df_copy["ClaimAmount"]<20000]
            summe = df_copy[prop].count()
            elem = list(df_copy[prop][np.invert(pd.isnull(df_copy[prop]))].to_numpy())
        else:
            # summe is the sum of the counts of each column ingredient
            summe = df[prop].count()
            # elem is the column elements
            elem = list(df[prop][np.invert(pd.isnull(df[prop]))].to_numpy())
            # create the plot
        plt.figure(figsize=(20, 20))
        plt.title("sum for " + str(prop) + " : " + str(summe), fontsize=40)
        # create the histogram
        if len(df[prop].unique()) <=25:
            if pd.isna(df[prop].unique()).any():
                nb_bins = len(df[prop].unique()) -1
            else:
                nb_bins = len(df[prop].unique())
        else:
            nb_bins= 10
        if prop == "ClaimAmount":
            nb_bins= 30
        plt.hist(elem, bins=nb_bins, histtype="bar", align='mid', color="darkblue", density=True)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=40)
        ax.tick_params(axis='both', which='minor', labelsize=40)
        ax.set_ylabel("density", fontsize=40, labelpad=10)
        ax.set_xlabel(prop, fontsize=40, labelpad=10)
        if prop in ["Region", "VehBrand"]:
            plt.xticks(rotation='vertical')
        # save plot
        outname = str(summe) + "x " + str(prop) + ".png"
        plt.savefig(os.path.join(countsplotpath, outname))
        plt.close("all")


def shap_plots(alg, y_pred, X_test, shap_values, plotpath):
    model_plotpath = os.path.join(plotpath, "shap")
    shap_file_name="shap"
    if 'Class' in str(alg):
        for n_claims_case in [0, 1]:
            plt.figure(figsize=(30, 30))
            example = np.where(y_pred == n_claims_case)[0][0]
            shap.plots.waterfall(shap_values[example], max_display=20, show=False)
            fig = plt.gcf()
            shapvalues_model_filename = str(alg) + shap_file_name + "_example_ypred_is" + str(n_claims_case) + '.png'
            shapfile = os.path.join(model_plotpath, shapvalues_model_filename)
            plt.savefig(shapfile)
            plt.close("all")
    plt.figure(figsize=(30, 30))
    shap.summary_plot(shap_values, X_test, show=False)
    fig = plt.gcf()
    shapsum_model_filename = str(alg) + shap_file_name + "summaryplot.png"
    shapfile = os.path.join(model_plotpath, shapsum_model_filename)
    plt.savefig(shapfile)
    plt.close("all")
    plt.figure(figsize=(30, 30))
    shap.plots.beeswarm(shap_values, show=False)
    fig = plt.gcf()
    shapbee_model_filename = str(alg) + shap_file_name + "beeswarmplot.png"
    shapfile = os.path.join(model_plotpath, shapbee_model_filename)
    plt.savefig(shapfile)
    plt.close("all")
    plt.figure(figsize=(30, 30))
    shap.plots.beeswarm(shap_values.abs, color="shap_red", show=False)
    fig = plt.gcf()
    shapredbee_model_filename = str(alg) + shap_file_name + "red_beeswarmplot.png"
    shapfile = os.path.join(model_plotpath, shapredbee_model_filename)
    plt.savefig(shapfile)
    plt.close("all")
    plt.figure(figsize=(30, 30))
    shap.plots.bar(shap_values, show=False)
    fig = plt.gcf()
    shapbar_model_filename = str(alg) + shap_file_name + "barplot.png"
    shapfile = os.path.join(model_plotpath, shapbar_model_filename)
    plt.savefig(shapfile)
    plt.close("all")
    plt.figure(figsize=(30, 30))
    shap.plots.bar(shap_values.abs.max(0), show=False)
    fig = plt.gcf()
    shapbarabs_model_filename = str(alg) + shap_file_name + "barplot_abs_max.png"
    shapabsfile = os.path.join(model_plotpath, shapbarabs_model_filename)
    plt.savefig(shapabsfile)
    plt.close("all")
    plt.figure(figsize=(30, 30))
    fig = plt.gcf()
    # shap.plots.scatter(shap_values[:, "DrivAge"], color=shap_values[:, "VehAge"], show=False)
    # will do
    for col in X_test.columns:
        plt.figure(figsize=(30, 30))
        shap.plots.scatter(shap_values[:, col], show=False)
        fig = plt.gcf()
        shapbarabs_model_filename = str(alg) + col + ".png"
        shapabsfile = os.path.join(model_plotpath, shapbarabs_model_filename)
        plt.savefig(shapabsfile)
        plt.close("all")
        # shap.plots.scatter(shap_values[:, "Density"])


def feature_importances_plot(feature_importances, plotpath, alg):
    plt.figure(figsize=(30, 30))
    feature_importances.plot.bar()
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=40)
    model_plotpath = os.path.join(plotpath, "models")
    feat_file_name="Feature_importances"
    features_model_filename = str(alg) + feat_file_name + '.png'
    featfile = os.path.join(model_plotpath, features_model_filename)
    plt.savefig(featfile)
    plt.close("all")


def confusion_matrix_and_class_report(alg, y_test, y_pred, plotpath, feat_sel, target):
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


def calculate_model_metrics(mod, X_train, y_train, target, X_test, y_test, alg):
    train_scores = mod.score(X_train, y_train, sample_weight=None)
    print("The Accuracy for training " + target + " by using " + str(alg) + " is: ", train_scores)
    test_scores = mod.score(X_test, y_test, sample_weight=None)
    print("The Accuracy for predicting " + target + " by using " + str(alg) + " is: ", test_scores)
    # precision = precision_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)