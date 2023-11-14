"""
script to compute model metrics
"""
from functools import partial
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_tweedie_deviance,
    accuracy_score,
    classification_report
    )
from analysis_plots import confusion_matrix_plot


def confusion_matrix_and_class_report(alg, y_test, y_pred, plotpath, feat_sel, target):
    """
    for classification models: plot the confusion matrix and return model metrics
    :input alg: algorithm, model
    :input y_test: pd.DataFrame of y_test data
    :input y_pred: pd.DataFrame of y_pred data
    :input plotplath: str of path where to store the plot
    :input feat_sel: list of str of selected features for model
    :input target: str of target property
    """
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


def calculate_model_metrics(mod, X_train, y_train, target, X_test, y_test):
    """
    calculate accuracy for train and test processes
    :input mod: model, method
    :iput X_train: pd.DataFrame of X_train data
    :input y_train: pd.DataFrame of y_train data
    :inpu target: str, target property
    :input X_test: pd.DataFrame of X_test data
    :input y_test: pd.DataFrame of y_test data
    """
    train_scores = mod.score(X_train, y_train, sample_weight=None)
    print("The Accuracy for training " + target + " by using " + str(mod) + " is: ", train_scores)
    test_scores = mod.score(X_test, y_test, sample_weight=None)
    print("The Accuracy for predicting " + target + " by using " + str(mod) + " is: ", test_scores)
    # precision = precision_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)


def score_estimator(
    estimator,
    X_train,
    X_test,
    df_train,
    df_test,
    target,
    weights,
    tweedie_powers=None,
):
    """Evaluate an estimator on train and test sets with different metrics"""

    metrics = [
        ("D²", None),  # Use default scorer if it exists
        ("mean abs. error", mean_absolute_error),
        ("mean squared error", mean_squared_error),
    ]
    if tweedie_powers:
        metrics += [
            (
                "mean Tweedie dev p={:.4f}".format(power),
                partial(mean_tweedie_deviance, power=power),
            )
            for power in tweedie_powers
        ]

    res = []
    for subset_label, X, df in [
        ("train", X_train, df_train),
        ("test", X_test, df_test),
    ]:
        y, _weights = df[target], df[weights]
        for score_label, metric in metrics:
            if isinstance(estimator, tuple) and len(estimator) == 2:
                # Score the model consisting of the product of frequency and
                # severity models.
                est_freq, est_sev = estimator
                y_pred = est_freq.predict(X) * est_sev.predict(X)
            else:
                y_pred = estimator.predict(X)

            if metric is None:
                if not hasattr(estimator, "score"):
                    continue
                score = estimator.score(X, y, sample_weight=_weights)
            else:
                score = metric(y, y_pred, sample_weight=_weights)

            res.append({"subset": subset_label, "metric": score_label, "score": score})

    res = (
        pd.DataFrame(res)
        .set_index(["metric", "subset"])
        .score.unstack(-1)
        .round(4)
        .loc[:, ["train", "test"]]
    )
    return res