"""
script to produce actual vs predicted plots for train and test datasets
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from global_vars import ACTUAL_VS_PREDICTED


def actual_pred_feature_plots(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        target,
        plotpath,
        weight
    ):
    """
    plot actual and predicted values for test and train datasets against a given feature,
    plots are saved
    :input model: str, modelname
    :input X_train: pd.Dataframe of X_train data
    :input y_train: pd.Dataframe of y_train data
    :input y_test: pd.Dataframe of y_test data
    :input: y_pred: pd.Dataframe of y_pred data
    :input target: str of target output for model
    :input plotpath: str of path to model-plot folder
    :input feat: str of feat for x-axis
    :input weight: str, colname of weights that should be applied
    """
    if ACTUAL_VS_PREDICTED:
        glm_model_plotpath = os.path.join(plotpath, "models")
        if any(ele in str(model) for ele in ['Poisson','Gamma']):
            glm_model_plotpath = os.path.join(glm_model_plotpath, "glm")
        elif 'RandomForest' in str(model):
            glm_model_plotpath = os.path.join(glm_model_plotpath, "rf")
        # restrict to non-sparse features
        for feat in ['Exposure', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']:
            if target != feat:
                modelname = str(model).split('(', maxsplit=1)[0] + "_" + feat + "_train_"
                glm_model_filename = modelname + str(target) + '.png'
                outfile = os.path.join(glm_model_plotpath, glm_model_filename)
                actual_pred_scatter_for_feature(
                    model,
                    X_train,
                    y_train,
                    target,
                    outfile,
                    feat,
                    weight
                )
                # print predict plots
                modelname = str(model).split('(', maxsplit=1)[0] + "_" + feat + "_predict_"
                glm_model_filename = modelname + str(target) + '.png'
                outfile = os.path.join(glm_model_plotpath, glm_model_filename)
                actual_pred_scatter_for_feature(
                    model,
                    X_test,
                    y_test,
                    target,
                    outfile,
                    feat,
                    weight
                )
                print(feat)
                plt.close('all')


def actual_pred_scatter_for_feature(model, y_test, y_pred, target, outfile, feat, weight):
    """
    plot actual and predicted values for test or train datasets against a given feature,
    plots are saved
    :input model: str, modelname
    :input y_test: pd.Dataframe of y_test data, will be labeled as 'actual'
    :input: y_pred: pd.Dataframe of y_pred data, labeled as'predicted'
    :input target: str of target output for model
    :input outfile: str of path to save the plot as png
    :input feat: str of feat for x-axis
    :input weight: str, colname of weights that should be applied
    """
    mdl = model
    fig = plt.figure(figsize=(25, 25))
    ax1 = fig.add_subplot(111)
    y_test_plt = y_test.copy()
    if not pd.isna(weight):
        y_pred_plt = mdl.predict(y_pred)
        y_pred_plt = pd.DataFrame(y_pred_plt, columns=[target])
        y_pred_plt[weight] = y_test[weight].to_list()
        y_pred_plt[target] = y_pred_plt[target]*y_pred_plt[weight]
        y_test_plt[target] = y_test_plt[target]*y_test[weight]
    else:
        y_pred_plt = mdl.predict(y_test)
        y_pred_plt = pd.DataFrame(y_pred_plt, columns=[target])
        y_test_plt[target] = y_pred
    y_pred_plt[feat] = y_test_plt[feat].to_list()
    y_test_plt = y_test_plt.groupby(feat, as_index=False).mean()
    y_pred_plt = y_pred_plt.groupby(feat, as_index=False).mean()
    ax1.plot(
        y_test_plt[feat],
        y_test_plt[target],
        c='b',
        marker="o",
        ls='',
        markersize=15,
        label='Actual'
    )
    ax1.plot(
        y_test_plt[feat],
        y_pred_plt[target],
        c='r',
        marker="o",
        ls='',
        markersize=10,
        label='Predicted'
        )
    plt.legend(loc="upper right", fontsize=40)
    ax1.set_ylabel("predicted", fontsize=40)
    ax1.set_xlabel("actual", fontsize=40)
    ax1.tick_params(axis='both', which='major', labelsize=40)
    ax1.tick_params(axis='both', which='minor', labelsize=40)
    # ax1.margins(50)
    if outfile:
        plt.savefig(outfile)
    plt.close("all")


def actual_pred_scatter(y_test, y_pred, target, plotpath, alg):
    """
    old version of actual_pred_scatter_for_feature, the difference is that
    here the predicted are on the x-axis.
    (TODO delete later?)
    """
    model_plotpath = os.path.join(plotpath, "models")
    modelname=str(alg)
    scatter_model_filename = modelname + str(target) + '.png'
    outfile = os.path.join(model_plotpath, scatter_model_filename)
    fig = plt.figure(figsize=(25, 25))
    ax1 = fig.add_subplot(111)
    ax1.scatter(y_test[target], y_pred[target], s=20, c='b', marker="o", label='RFRegressor')
    plt.legend(loc="upper right", fontsize=40)
    ax1.set_ylabel("predicted", fontsize=40)
    ax1.set_xlabel("actual", fontsize=40)
    ax1.tick_params(axis='both', which='major', labelsize=40)
    ax1.tick_params(axis='both', which='minor', labelsize=40)
    ax1.axline((0, 0), slope=0.5, c='black')
    ax1.set_xlim(0, 5000)
    ax1.set_ylim(0, 5000)
    plt.savefig(outfile)
    plt.close("all")
