from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from strlearn.metrics import recall, specificity
from scipy.spatial.distance import cdist


class OCWE(ClassifierMixin, BaseEnsemble):

    def __init__(self,
                 n_estimators=10,
                 cluster_method=MiniBatchKMeans,
                 cluster_metric=silhouette_score,
                 alpha=1,
                 beta=1,
                 gamma=1,
                 delta=1,
                 kmax=10,
                 nu_min=0.1,
                 nu_maj=0.1,
                 stacker=LogisticRegression()):

        self.n_estimators = n_estimators
        self.cluster_method = cluster_method
        self.cluster_metric = cluster_metric
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.kmax = kmax
        self.nu_min = nu_min
        self.nu_maj = nu_maj
        self.stacker = stacker

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
            self.cluster_means_maj = []
            self.cluster_means_min = []
            self.drift_detector = None
            self.metrics_array = []

        # ________________________________________
        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # ________________________________________
        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # ________________________________________
        # Find minority and majority names
        if not hasattr(self, "minority_name") or not hasattr(self, "majority_name"):
            self.minority_name, self.majority_name = self.minority_majority_name(
                y)

        # ________________________________________
        # Prune minority
        to_delete = []
        for i, w in enumerate(self.iter_min):
            if w > self.n_estimators:
                to_delete.append(i)
            self.iter_min[i] += 1
        to_delete.reverse()
        for i in to_delete:
            del self.iter_min[i]
            del self.ensemble_min[i]
            del self.cluster_means_min[i]

        # ________________________________________
        # Prune majority
        to_delete = []
        for i, w in enumerate(self.iter_maj):
            if w > self.n_estimators:
                to_delete.append(i)
            self.iter_maj[i] += 1
        to_delete.reverse()
        for i in to_delete:
            del self.iter_maj[i]
            del self.ensemble_maj[i]
            del self.cluster_means_maj[i]

        # ________________________________________
        # Split data
        minority, majority = self.minority_majority_split(
            X, y, self.minority_name, self.majority_name)

        # ________________________________________
        # Check imbalance ratio
        if len(majority) > len(minority):
            # ________________________________________
            # Cluster as minority
            samples, cluster_means, n_clusters = self._best_number_of_clusters(
                minority, self.kmax)
            for i in range(n_clusters):
                if len(samples[i]) != 0:
                    self.ensemble_min.append(
                        clone(OneClassSVM(nu=self.nu_min, kernel="rbf")).fit(samples[i]))
                    self.iter_min.append(0)
                    self.cluster_means_min.append(cluster_means[i])

            # ________________________________________
            # Cluster as majority
            samples, cluster_means, n_clusters = self._clustering(
                majority, int(len(majority) / len(minority)))
            for i in range(n_clusters):
                if len(samples[i]) != 0:
                    self.ensemble_maj.append(
                        clone(OneClassSVM(nu=self.nu_maj, kernel="rbf")).fit(samples[i]))
                    self.iter_maj.append(0)
                    self.cluster_means_maj.append(cluster_means[i])

        else:
            # ________________________________________
            # Cluster as majority
            samples, cluster_means, n_clusters = self._clustering(
                minority, int(len(minority) / len(majority)))
            for i in range(n_clusters):
                if len(samples[i]) != 0:
                    self.ensemble_min.append(
                        clone(OneClassSVM(nu=self.nu_min, kernel="rbf")).fit(samples[i]))
                    self.iter_min.append(0)
                    self.cluster_means_min.append(cluster_means[i])

            # ________________________________________
            # Cluster as minority
            samples, cluster_means, n_clusters = self._best_number_of_clusters(
                majority, self.kmax)
            for i in range(n_clusters):
                if len(samples[i]) != 0:
                    self.ensemble_maj.append(
                        clone(OneClassSVM(nu=self.nu_maj, kernel="rbf")).fit(samples[i]))
                    self.iter_maj.append(0)
                    self.cluster_means_maj.append(cluster_means[i])

        # ________________________________________
        # Calculate IR weights
        irw_min = 1 - (len(minority) / len(X))
        irw_maj = 1 - (len(majority) / len(X))

        # ________________________________________
        # Calculate distance weights
        distances_min = self.calculate_distances(
            self.cluster_means_min, self.cluster_means_maj)
        distances_maj = self.calculate_distances(
            self.cluster_means_maj, self.cluster_means_min)

        # ________________________________________
        # Calculate score weights
        scores_min, scores_maj = self.calculate_scores(X, y)
        # print("scores_maj", scores_maj)

        # ________________________________________
        # Calculate age weights
        age_min, age_maj = self.calculate_ages()

        # ________________________________________
        # Combine all weights
        self.weights_min = np.nan_to_num(self.alpha * distances_min + self.beta * \
            scores_min + self.gamma * age_min + self.delta * irw_min)
        self.weights_maj = np.nan_to_num(self.alpha * distances_maj + self.beta * \
            scores_maj + self.gamma * age_maj + self.delta * irw_maj)
        # print("dist", distances_min)
        # print("age", age_min)
        # print("ir", irw_min)
        # print("scr", scores_min)

        # ________________________________________
        # Stacking
        probas_min = [clf.decision_function(
            X) * w_min for clf, w_min in zip(self.ensemble_min, self.weights_min)]
        probas_maj = [clf.decision_function(
            X) * w_maj for clf, w_maj in zip(self.ensemble_maj, self.weights_maj)]

        probas_ = np.concatenate([probas_min, probas_maj], axis=0).T

        uq, cnt = np.unique(y, return_counts=True)
        weights_ = {uq[0]:cnt[0], uq[1]:cnt[1]}
        # print(uq, cnt)
        self.stacker.class_weight = "balanced"
        # print("probas", probas_)
        self.stacker = self.stacker.fit(probas_, y)




    def calculate_distances(self, clusters1, clusters2):
        dist = cdist(clusters1, clusters2, "euclidean")
        dist = np.min(dist, axis=1)
        dist = dist / np.linalg.norm(dist)

        return dist

    def calculate_scores(self, X, y):
        scores_maj = []
        for clf in self.ensemble_maj:
            pred = clf.predict(X)
            m = pred == 1
            pred[pred == 1] = 0
            pred[pred == -1] = 1
            scores_maj.append(specificity(pred[m], y[m]))

        scores_min = []
        for clf in self.ensemble_min:
            pred = clf.predict(X)
            m = pred == 1
            pred[pred == 1] = 1
            pred[pred == -1] = 0
            scores_min.append(recall(pred[m], y[m]))

        scores_min = scores_min / np.linalg.norm(scores_min)
        scores_maj = scores_maj / np.linalg.norm(scores_maj)

        return scores_min, scores_maj

    def calculate_ages(self):
        age_maj = np.array(self.iter_maj)
        age_min = np.array(self.iter_min)

        age_maj = 1 - age_maj / np.linalg.norm(age_maj)
        age_min = 1 - age_min / np.linalg.norm(age_min)

        age_maj = np.nan_to_num(age_maj, nan=1)
        age_min = np.nan_to_num(age_min, nan=1)

        return age_min, age_maj

    def _clustering(self, data, k):
        cluster_model = self.cluster_method(n_clusters=k)
        labels = cluster_model.fit_predict(data)
        clusters = labels

        n_clusters = k
        samples = [[] for i in range(n_clusters)]

        for i, x in enumerate(clusters):
            samples[x].append(data[i].tolist())

        if hasattr(cluster_model, "cluster_centers_"):
            cluster_centers = cluster_model.cluster_centers_
        else:
            cluster_centers = []
            for sp in samples:
                cluster_centers.append(np.mean(sp, axis=0))
            cluster_centers = np.array(cluster_centers)

        return samples, cluster_centers, n_clusters

    def _best_number_of_clusters(self, data, kmax=10):
        score_vals = []
        clusters = []
        cluster_models = []
        if len(data) == 0:
            return None, None, 0
        for k in range(1, kmax + 1):
            try:
                cluster_model = self.cluster_method(n_clusters=k)
                labels = cluster_model.fit_predict(data)
                clusters.append(labels)
                cluster_models.append(cluster_model)

                if k == 1:
                    score_vals.append(0)
                else:
                    score_vals.append(self.cluster_metric(data, labels))
            except Exception as ex:
                # print(ex)
                break

        best_number = np.argmax(np.array(score_vals))
        n_clusters = best_number + 1
        samples = [[] for i in range(n_clusters)]

        for i, x in enumerate(clusters[best_number]):
            samples[x].append(data[i].tolist())

        if hasattr(cluster_models[best_number], "cluster_centers_"):
            cluster_centers = cluster_models[best_number].cluster_centers_
        else:
            cluster_centers = []
            for sp in samples:
                cluster_centers.append(np.mean(sp, axis=0))
            cluster_centers = np.array(cluster_centers)

        return samples, cluster_centers, n_clusters

    def predict(self, X):
        # ________________________________________
        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        probas_min = [clf.decision_function(
            X) * w_min for clf, w_min in zip(self.ensemble_min, self.weights_min)]
        probas_maj = [clf.decision_function(
            X) * w_maj for clf, w_maj in zip(self.ensemble_maj, self.weights_maj)]

        probas_ = np.concatenate([probas_min, probas_maj], axis=0).T

        maj = self.stacker.predict(probas_)

        # maj = np.argmax(self.predict_proba(X), axis=1)
        return maj

    def predict_proba(self, X):
        probas_min = np.max([clf.decision_function(
            X) * w_min for clf, w_min in zip(self.ensemble_min, self.weights_min)], axis=0)
        probas_maj = np.max([clf.decision_function(
            X) * w_maj for clf, w_maj in zip(self.ensemble_maj, self.weights_maj)], axis=0)

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
        minority_name : object
            Name of minority class.
        majority_name : object
            Name of majority class.

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

    def get_params(self, deep=True):

        return {"n_estimators": self.n_estimators,
                "cluster_method": self.cluster_method,
                "cluster_metric": self.cluster_metric,
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "delta": self.delta}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
