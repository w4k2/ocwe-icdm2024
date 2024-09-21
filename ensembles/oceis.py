from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score


class OCEIS(ClassifierMixin, BaseEnsemble):

    def __init__(self, base_estimator=OneClassSVM(nu=0.1), n_estimators=10, cluster_method=KMeans):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.cluster_method = cluster_method

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

        # Find minority and majority names
        if not hasattr(self, "minority_name") or not hasattr(self, "majority_name"):
            self.minority_name, self.majority_name = self.minority_majority_name(y)

        # Prune minority
        to_delete = []
        for i, w in enumerate(self.iter_min):
            if w <= 0:
                to_delete.append(i)
            self.iter_min[i] -= 1
        to_delete.reverse()
        for i in to_delete:
            del self.iter_min[i]
            del self.ensemble_min[i]

        # Prune majority
        to_delete = []
        for i, w in enumerate(self.iter_maj):
            if w <= 0:
                to_delete.append(i)
            self.iter_maj[i] -= 1
        to_delete.reverse()
        for i in to_delete:
            del self.iter_maj[i]
            del self.ensemble_maj[i]

        # Split data
        minority, majority = self.minority_majority_split(X, y, self.minority_name, self.majority_name)

        samples, n_of_clust = self._best_number_of_clusters(minority, 10)

        for i in range(n_of_clust):
            self.ensemble_min.append(clone(self.base_estimator).fit(samples[i]))
            self.iter_min.append(self.n_estimators)

        samples, n_of_clust = self._best_number_of_clusters(majority, 10)
        for i in range(n_of_clust):
            self.ensemble_maj.append(clone(self.base_estimator).fit(samples[i]))
            self.iter_maj.append(self.n_estimators)

        return self

    def _best_number_of_clusters(self, data, kmax=10):

        sil_values = []
        clusters = []

        for k in range(2, kmax+1):
            try:
                cluster_model = self.cluster_method(n_clusters=k)
                labels = cluster_model.fit_predict(data)
                clusters.append(labels)
                sil_values.append(silhouette_score(data, labels, metric='euclidean'))
            except Exception:
                break

        best_number = np.argmax(np.array(sil_values))
        n_of_clust = best_number+2
        samples = [[] for i in range(n_of_clust)]

        for i, x in enumerate(clusters[best_number]):
            samples[x].append(data[i].tolist())

        return samples, n_of_clust

    def predict(self, X):
        # ________________________________________
        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        maj = np.argmax(self.predict_proba(X), axis=1)
        return maj

    def predict_proba(self, X):
        probas_min = np.max([clf.decision_function(X) for clf in self.ensemble_min], axis=0)
        probas_maj = np.max([clf.decision_function(X) for clf in self.ensemble_maj], axis=0)
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
