from classes.Models import LinearRegressionModel, KNRegressionModel, KNClassifierModel, NBModel, SVCModel
from matplotlib import pyplot as plt
import itertools
import numpy as np


def lr(df, all_features, current_guess):
    model = LinearRegressionModel(df, all_features, current_guess)
    model.fit()
    model.show_score()


def knr(df, all_features, current_guess, k, plot_best_k=1):
    model = KNRegressionModel(df, all_features, current_guess)
    if plot_best_k > 1:
        model.plot_best_k(1, plot_best_k)

    model.fit(k)
    model.show_score()


def knc(df, all_features, current_guess, k, show_report=True, plot_best_k=1, show_matrix=False, matrix_classes=[0, 1]):
    model = KNClassifierModel(df, all_features, current_guess)
    if plot_best_k > 1:
        model.plot_best_k(1, plot_best_k)
    model.fit(k, show_report)
    model.show_score()

    if show_report:
        model.show_report()

        if len(model.matrix) > 0 and show_matrix:
            show_confusion_matrix(model.matrix, matrix_classes=matrix_classes)


def nbc(df, all_features, current_guess, show_report=True, show_matrix=False, matrix_classes=[0, 1]):
    model = NBModel(df, all_features, current_guess)
    model.fit(show_report=show_report)
    model.show_score()

    if show_report:
        model.show_report()

        if len(model.matrix) > 0 and show_matrix:
            show_confusion_matrix(model.matrix, matrix_classes=matrix_classes)


def svc(df, all_features, current_guess, vector_kernel='rbf', vector_c=1, vector_gamma=1,
        show_report=True, best_accuracy=False, show_matrix=False, matrix_classes=[0, 1]):
    model = SVCModel(df, all_features, current_guess)
    if best_accuracy:
        model.compute_best_accuracy(vector_kernel=vector_kernel, v_from=1, v_to=11)

    model.fit(vector_kernel=vector_kernel, vector_c=vector_c, vector_gamma=vector_gamma, show_report=show_report)
    model.show_score()

    if show_report:
        model.show_report()

        if len(model.matrix) > 0 and show_matrix:
            show_confusion_matrix(model.matrix, matrix_classes=matrix_classes)


def show_confusion_matrix(cm, matrix_classes):
    plt.figure()
    plot_confusion_matrix(cm, classes=matrix_classes)
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
