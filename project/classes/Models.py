from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC


class Model:
    model = ''
    score = 0
    train_test_split = 0
    x_train, x_test, y_train, y_test = 0, 0, 0, 0

    def __init__(self, data, features_list, guess):
        data = data.dropna(subset=["height"])
        self.guess = data.loc[:, guess]
        self.features = data.loc[:, features_list]

    def prepare_train_test_split(self):
        x, y = self.normalize_data(self.features, self.guess)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        self.train_test_split = 1

        return x_train, x_test, y_train, y_test

    def show_score(self):
        print(self.score)


class RegressionModel(Model):
    def normalize_data(self, x, y):
        min_max_scaler = preprocessing.MinMaxScaler()

        x_scaled = min_max_scaler.fit_transform(x.values)
        x = pd.DataFrame(x_scaled, columns=x.columns)

        y_scaled = min_max_scaler.fit_transform(y.values)
        y = pd.DataFrame(y_scaled, columns=y.columns)

        return x, y


class ClassificationModel(Model):
    report = ''
    matrix = ''

    def normalize_data(self, x, y):
        min_max_scaler = preprocessing.MinMaxScaler()

        x_scaled = min_max_scaler.fit_transform(x.values)
        x = pd.DataFrame(x_scaled, columns=x.columns)

        return x, y

    def show_report(self):
        print(self.report)
        # print(self.matrix)

    def build_report(self, classifier_model):
        y_predicted = classifier_model.predict(self.x_test)
        report = classification_report(self.y_test, y_predicted)
        matrix = confusion_matrix(self.y_test, y_predicted)

        return report, matrix


class LinearRegressionModel(RegressionModel):
    def fit(self):
        x_train, x_test, y_train, y_test = self.prepare_train_test_split()
        model = LinearRegression()
        model.fit(x_train, np.ravel(y_train, order='C'))

        self.score = model.score(x_test, y_test)


class SVCModel(ClassificationModel):
    def fit(self, vector_kernel='rbf', vector_degree=3, vector_gamma=1, vector_c=1, show_report=False):
        if self.train_test_split == 0:
            self.x_train, self.x_test, self.y_train, self.y_test = self.prepare_train_test_split()

        model = SVC(kernel=vector_kernel, degree=vector_degree, gamma=vector_gamma, C=vector_c)
        model.fit(self.x_train, np.ravel(self.y_train, order='C'))

        self.score = model.score(self.x_test, self.y_test)

        if show_report:
            self.report, self.matrix = self.build_report(model)

    def compute_best_accuracy(self, vector_kernel, v_from, v_to):
        best = {'score': 0, 'gamma': 0, 'c': 0}

        for g in range(v_from, v_to):
            for c in range(1, 5):
                self.fit(vector_kernel, vector_gamma=g, vector_c=c)
                if self.score > best['score']:
                    best['score'] = self.score
                    best['gamma'] = g
                    best['c'] = c

                print('%d %d %f' % (g, c, self.score))
        print("Best Score:")
        print(best)


class NBModel(ClassificationModel):
    def fit(self, show_report=False):
        if self.train_test_split == 0:
            self.x_train, self.x_test, self.y_train, self.y_test = self.prepare_train_test_split()

        model = MultinomialNB()
        model.fit(self.x_train, np.ravel(self.y_train, order='C'))

        self.score = model.score(self.x_test, self.y_test)

        if show_report:
            self.report, self.matrix = self.build_report(model)


class KNModel(Model):
    k = 1
    report = ''

    def fit(self, k):
        pass

    def load_model(self, k):
        pass

    def show_score(self):
        print('%d -> %f' % (self.k, self.score))

    def plot_best_k(self, k_from, k_to):
        accuracies = []
        best = {'score': 0, 'k': 0}

        for k in range(k_from, k_to):
            self.fit(k)
            accuracies.append(self.score)
            self.show_score()
            if self.score > best['score']:
                best['score'] = self.score
                best['k'] = k

        print("Best k:")
        print(best)

        k_list = [i for i in range(k_from, k_to)]

        plt.plot(k_list, accuracies)
        plt.xlabel("k")
        plt.ylabel("Validation Accuracy")
        plt.show()


class KNRegressionModel(KNModel, RegressionModel):
    def fit(self, k):
        if self.train_test_split == 0:
            self.x_train, self.x_test, self.y_train, self.y_test = self.prepare_train_test_split()

        model = self.load_model(k)
        model.fit(self.x_train, np.ravel(self.y_train, order='C'))

        self.score = model.score(self.x_test, self.y_test)
        self.k = k

    def load_model(self, k):
        return KNeighborsRegressor(n_neighbors=k, weights="uniform")


class KNClassifierModel(KNModel, ClassificationModel):
    def fit(self, k, show_report=False):
        if self.train_test_split == 0:
            self.x_train, self.x_test, self.y_train, self.y_test = self.prepare_train_test_split()

        model = self.load_model(k)
        model.fit(self.x_train, np.ravel(self.y_train, order='C'))

        self.score = model.score(self.x_test, self.y_test)
        self.k = k

        if show_report:
            self.report, self.matrix = self.build_report(model)

    def load_model(self, k):
        return KNeighborsClassifier(n_neighbors=k)
