from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score, accuracy_score

from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import KMeansSMOTE


class WECOI(ClassifierMixin, BaseEnsemble):

    def __init__(self, base_estimator=OneClassSVM(nu=0.1), tau=10, threshold=1, hard_voting=True):
        self.base_estimator = base_estimator
        self.tau = tau
        self.threshold = threshold
        self.hard_voting = hard_voting

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)

        if not hasattr(self, "ensemble_maj"):
            self.ensemble_maj = []
            self.ensemble_min = []
            self.iter_maj = []
            self.iter_min = []
            self.weights_min = []
            self.weights_maj = []
        else:
            for i, w in enumerate(self.iter_min):
                self.iter_min[i] = w+1

            for i, w in enumerate(self.iter_maj):
                self.iter_maj[i] = w+1

        # Check if is more than one class
        if len(np.unique(y)) == 1:
            raise ValueError("Only one class in data chunk.")

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        self.n_classes = len(self.classes_)

        # Find minority and majority names
        if not hasattr(self, "minority_name") or not hasattr(self, "majority_name"):
            self.minority_name, self.majority_name = self.minority_majority_name(y)


        # Prune minority
        if len(self.ensemble_maj) >= self.tau:
            self.weights_min = []
            for idx, clf in enumerate(self.ensemble_min):
                y_pred = clf.predict(X)
                y_pred[y_pred==1] = self.minority_name
                y_pred[y_pred==-1] = self.majority_name
                score = accuracy_score(y, y_pred)
                self.weights_min.append(score/self.iter_min[idx])
            
            worst_min = np.argmin(self.weights_min)
            del self.weights_min[worst_min]
            del self.ensemble_min[worst_min]
            del self.iter_min[worst_min]

        if len(self.ensemble_min) >= self.tau:
            self.weights_maj = []
            for idx, clf in enumerate(self.ensemble_maj):
                y_pred = clf.predict(X)
                y_pred[y_pred==1] = self.majority_name
                y_pred[y_pred==-1] = self.minority_name
                score = accuracy_score(y, y_pred)
                self.weights_maj.append(score/self.iter_maj[idx])
            
            worst_maj = np.argmin(self.weights_maj)
            del self.weights_maj[worst_maj]
            del self.ensemble_maj[worst_maj]
            del self.iter_maj[worst_maj]

        # Split data
        minority, majority = self.minority_majority_split(X, y, self.minority_name, self.majority_name)

        cnn = EditedNearestNeighbours()
        maj_l = [self.majority_name for x in range(majority.shape[0])]
        maj_l.append(self.minority_name)
        majority = np.append(majority, minority[0:1], axis=0)
        new_majority = cnn.fit_resample(majority, maj_l)[0]

        if (minority.shape[0]/(new_majority.shape[0]+minority.shape[0])) > self.threshold:
            kms = KMeansSMOTE(cluster_balance_threshold=0.01, k_neighbors=2)
            new_X = np.concatenate((new_majority, minority), axis=0)
            new_y = [self.majority_name for x in range(new_majority.shape[0])].extend([self.majority_name for x in range(minority.shape[0])])
            new_X, new_y = kms.fit_resample(X, y)
            new_minority, _ = self.minority_majority_split(new_X, new_y, self.minority_name, self.majority_name)
        else:
            new_minority = minority


        # Train minority
        clf_min = clone(self.base_estimator).fit(new_minority)
        self.ensemble_min.append(clf_min)
        self.iter_min.append(1)
        y_pred = clf_min.predict(X)
        y_pred[y_pred==1] = self.minority_name
        y_pred[y_pred==-1] = self.majority_name
        score = accuracy_score(y, y_pred)
        self.weights_min.append(score)

        # Train majority
        clf_maj = clone(self.base_estimator).fit(new_majority)
        self.ensemble_maj.append(clf_maj)
        self.iter_maj.append(1)
        y_pred = clf_maj.predict(X)
        y_pred[y_pred==1] = self.majority_name
        y_pred[y_pred==-1] = self.minority_name
        score = accuracy_score(y, y_pred)
        self.weights_maj.append(score)

        return self

    def predict(self, X):


        # ________________________________________
        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        probas_min = []
        probas_min.append(np.max([clf.decision_function(X) for clf in self.ensemble_min], axis=0))
        probas_min = np.array(probas_min)
        
        probas_maj = []
        probas_maj.append(np.max([clf.decision_function(X) for clf in self.ensemble_maj], axis=0))
        probas_maj = np.array(probas_maj)

        probas_ = []
        for p_min, p_maj in zip(probas_min, probas_maj):
            probas_.append(np.stack((p_maj, p_min), axis=1))

        if self.hard_voting:
            predictions = np.argmax(probas_, axis=2)
            maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions.T)
        else:
            probas_mean = np.average(probas_, axis=0)
            maj = np.argmax(probas_mean, axis=1)

        # maj = np.argmax(self.predict_proba(X), axis=1)
        return maj

    def predict_proba(self, X):
        probas_min = np.max([clf.decision_function(X)*wght for wght, clf in zip(self.weights_min, self.ensemble_min)], axis=0)
        probas_maj = np.max([clf.decision_function(X)*wght for wght, clf in zip(self.weights_maj, self.ensemble_maj)], axis=0)
        probas_ = np.stack((probas_maj, probas_min), axis=1)

        return probas_

    def minority_majority_split(self, X, y, minority_name, majority_name):
        """Returns minority and majority data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        minority : array-like, shape = [n_samples, n_features]
            Minority class samples.
        majority : array-like, shape = [n_samples, n_features]
            Majority class samples.
        """

        minority_ma = np.ma.masked_where(y == minority_name, y)
        minority = X[minority_ma.mask]

        majority_ma = np.ma.masked_where(y == majority_name, y)
        majority = X[majority_ma.mask]

        return minority, majority

    def minority_majority_name(self, y):
        """Returns the name of minority and majority class

        Parameters
        ----------
        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        minority_name : object
            Name of minority class.
        majority_name : object
            Name of majority class.
        """

        unique, counts = np.unique(y, return_counts=True)

        if counts[0] > counts[1]:
            majority_name = unique[0]
            minority_name = unique[1]
        else:
            majority_name = unique[1]
            minority_name = unique[0]

        return minority_name, majority_name
