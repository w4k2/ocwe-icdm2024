from ensembles import OCWE
import strlearn as sl

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import os
import warnings
warnings.simplefilter("ignore")


def main():
    directory = "param_setup/"
    mypath = "streams/%s" % directory
    streams = ["%s%s" % (mypath, f) for f in os.listdir(
        mypath) if os.path.isfile(os.path.join(mypath, f))]

    chunk_size = 200
    train_chunks = 20
    test_chunks = 5
    resolution = 0.1
    steps = int(1 / resolution) + 1

    metric_names = ["g_mean", "f1_score", "recall", "specificity"]

    for stream_name in streams:

        stream = sl.streams.ARFFParser(stream_name, chunk_size, train_chunks+test_chunks)
        X_array = []
        y_array = []
        for _ in range(train_chunks):
            X, y = stream.get_chunk()
            X_array.append(X)
            y_array.append(y)

        X_test_array = []
        y_test_array = []
        for _ in range(test_chunks):
            X, y = stream.get_chunk()
            X_test_array.append(X)
            y_test_array.append(y)

        # -----------------------
        # -  ALPHA BETA
        # -----------------------
        metrics_array = np.zeros((4, steps, steps))
        for idx, p1 in enumerate(tqdm(np.around(np.arange(0, 1.1, resolution), 1))):
            for jdx, p2 in enumerate(tqdm(np.around(np.arange(0, 1.1, resolution), 1))):
                clf = OCWE(alpha=p1, beta=p2)
                for X, y in zip(X_array, y_array):
                    clf.partial_fit(X, y)

                geometric_mean_score_1_array = []
                f1_score_array = []
                recall_array = []
                specificity_array = []

                for X, y in zip(X_array, y_array):
                    y_pred = clf.predict(X)
                    geometric_mean_score_1_array.append(sl.metrics.geometric_mean_score_1(y_pred, y))
                    f1_score_array.append(sl.metrics.f1_score(y_pred, y))
                    recall_array.append(sl.metrics.recall(y_pred, y))
                    specificity_array.append(sl.metrics.specificity(y_pred, y))

                metrics_array[0, jdx, idx] = np.mean(geometric_mean_score_1_array)
                metrics_array[1, jdx, idx] = np.mean(f1_score_array)
                metrics_array[2, jdx, idx] = np.mean(recall_array)
                metrics_array[3, jdx, idx] = np.mean(specificity_array)

        save_grid_results("Alpha", "Beta", metrics_array,
                          metric_names, stream_name)
        plot_grid_results("Alpha", "Beta", metrics_array,
                          metric_names, stream_name, resolution)

        # metrics_array = load_data(stream_name, metric_names, steps, "Alpha", "Beta")
        plot_grid_results("Alpha", "Beta", metrics_array,
                          metric_names, stream_name, resolution)

        # -----------------------
        # -  GAMMA DELTA
        # -----------------------
        metrics_array = np.zeros((4, steps, steps))
        for idx, p1 in enumerate(tqdm(np.around(np.arange(0, 1.1, resolution), 1))):
            for jdx, p2 in enumerate(tqdm(np.around(np.arange(0, 1.1, resolution), 1))):
                clf = OCWE(gamma=p1, delta=p2)
                for X, y in zip(X_array, y_array):
                    clf.partial_fit(X, y)

                geometric_mean_score_1_array = []
                f1_score_array = []
                recall_array = []
                specificity_array = []

                for X, y in zip(X_array, y_array):
                    y_pred = clf.predict(X)
                    geometric_mean_score_1_array.append(sl.metrics.geometric_mean_score_1(y_pred, y))
                    f1_score_array.append(sl.metrics.f1_score(y_pred, y))
                    recall_array.append(sl.metrics.recall(y_pred, y))
                    specificity_array.append(sl.metrics.specificity(y_pred, y))

                metrics_array[0, jdx, idx] = np.mean(geometric_mean_score_1_array)
                metrics_array[1, jdx, idx] = np.mean(f1_score_array)
                metrics_array[2, jdx, idx] = np.mean(recall_array)
                metrics_array[3, jdx, idx] = np.mean(specificity_array)

        save_grid_results("Gamma", "Delta", metrics_array,
                          metric_names, stream_name)
        plot_grid_results("Gamma", "Delta", metrics_array,
                          metric_names, stream_name, resolution)

        # metrics_array = load_data(stream_name, metric_names, steps, "Gamma", "Delta")
        plot_grid_results("Gamma", "Delta", metrics_array,
                          metric_names, stream_name, resolution)


def save_grid_results(param_1, param_2, metrics_array, metric_names, stream_name):
    for idx, metric in enumerate(metric_names):

        filename = ("%s%s_" % (param_1[0], param_2[0])) + \
            metric + "__" + (stream_name.split("/")[-1])[0:-5]
        filepath = "results/param_setup_grid/vals/%s" % filename

        if not os.path.exists("results/param_setup_grid/vals"):
            os.makedirs("results/param_setup_grid/vals")

        np.save(filepath, metrics_array[idx, :, :])
        np.savetxt(filepath + ".csv",
                   metrics_array[idx, :, :],
                   delimiter=";",
                   fmt="%0.3f")


def plot_grid_results(param_1, param_2, metrics_array, metric_names, stream_name, resolution):
    for idx, metric in enumerate(metric_names):

        fig, ax = plt.subplots(figsize=(7, 7))
        mean = np.mean(metrics_array[idx, :, :])
        max = np.max(metrics_array[idx, :, :])
        ax.imshow(metrics_array[idx, :, :], cmap='Greys')
        plt.xlabel(param_1)
        plt.ylabel(param_2)

        labels = np.around(np.arange(0, 1.1, resolution), 1)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if metrics_array[idx, i, j] > mean:
                    if metrics_array[idx, i, j] == max:
                        ax.text(j, i,
                                '%.3f' % metrics_array[idx, i, j],
                                ha="center",
                                va="center",
                                color="w",
                                fontweight="bold",
                                fontsize="11")
                        continue
                    ax.text(j, i,
                            '%.3f' % metrics_array[idx, i, j],
                            ha="center",
                            va="center",
                            color="w")
                else:
                    ax.text(j, i,
                            '%.3f' % metrics_array[idx, i, j],
                            ha="center",
                            va="center",
                            color="black")

        if not os.path.exists("results/param_setup_grid/plots"):
            os.makedirs("results/param_setup_grid/plots")

        filename = ("%s%s_" % (param_1[0], param_2[0])) + \
            metric + "__" + (stream_name.split("/")[-1])[0:-5]

        plt.tight_layout()
        plt.savefig("results/param_setup_grid/plots/%s.png" % filename)
        plt.savefig("results/param_setup_grid/plots/%s.eps" % filename)

        filepath = "results/param_setup_grid/vals/%s" % filename

        if not os.path.exists("results/param_setup_grid/vals"):
            os.makedirs("results/param_setup_grid/vals")

        np.save(filepath, metrics_array[idx, :, :])
        np.savetxt(filepath + ".csv",
                   metrics_array[idx, :, :],
                   delimiter=";",
                   fmt="%0.3f")


def load_data(stream_name, metric_names, steps, param_1, param_2):
    metrics_array = np.zeros((4, steps, steps))
    for idx, metric in enumerate(metric_names):
        filename = ("%s%s_" % (param_1[0], param_2[0])) + \
            metric + "__" + (stream_name.split("/")[-1])[0:-5]
        filepath = "results/param_setup_grid/vals/%s" % filename
        metrics_array[idx, :, :] = np.load(filepath+".npy")
    return metrics_array


if __name__ == '__main__':
    main()
