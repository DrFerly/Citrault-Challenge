import shap
import os
import numpy as np
import matplotlib.pyplot as plt


def shap_plots(alg, y_pred, X_test, shap_values, plotpath):
    model_plotpath = os.path.join(plotpath, "shap")
    shap_file_name="shap"
    if 'Class' in str(alg):
        for n_claims_case in [0, 1]:
            plt.figure(figsize=(30, 30))
            example = np.where(y_pred == n_claims_case)[0][0]
            shap.plots.waterfall(shap_values[example], max_display=20, show=False)
            fig = plt.gcf()
            shapvalues_model_filename = str(alg) + shap_file_name + \
                "_example_ypred_is" + str(n_claims_case) + '.png'
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
    # create shapley value plots:
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
    