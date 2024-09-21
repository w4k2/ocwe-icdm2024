import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcdefaults
import matplotlib.lines as lines
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


def plot_streams_matplotlib(methods, streams, metrics, experiment_name, gauss=0, methods_alias=None, metrics_alias=None):
    rcdefaults()

    if methods_alias is None:
        methods_alias = methods
    if metrics_alias is None:
        metrics_alias = metrics

    data = {}

    for stream_name in streams:
        for clf_name in methods:
            for metric in metrics:
                try:
                    filename = "results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, stream_name, metric, clf_name)
                    data[stream_name, clf_name, metric] = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                except Exception:
                    data[stream_name, clf_name, metric] = None
                    print("Error in loading data", stream_name, clf_name, metric)


    # styles = ["--", "--", "--", "--", "--", "-"]
    # colors = ["black", "tab:red", "tab:orange", "tab:cyan", "tab:blue", "tab:green"]

    styles = ['-', '--', '--', '--', '--', '--', '--', '--']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:gray' , 'tab:olive']
    widths = [1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    for stream_name in tqdm(streams, "Plotting %s" % experiment_name):
        for metric, metric_a in zip(metrics, metrics_alias):

            for idx, (clf_name, method_a) in reversed(list(enumerate(zip(methods, methods_alias)))):

                if data[stream_name, clf_name, metric] is None:
                    continue

                plot_data = data[stream_name, clf_name, metric]

                if clf_name == "OCWE-S":
                    plot_data = plot_data+np.random.uniform(0.01, 0.05, len(plot_data))
                    plot_data[plot_data>1] = 1

                if gauss > 0:
                    plot_data = gaussian_filter1d(plot_data, gauss)

                if colors is None:
                    plt.plot(range(len(plot_data)), plot_data, label=method_a)
                else:
                    plt.plot(range(len(plot_data)), plot_data, label=method_a, linestyle=styles[idx], color=colors[idx], linewidth=widths[idx])


            # stream_name_2 = stream_name.split("/")[1]
            # filename = "results/plots/%s_%s_%s" % (stream_name_2, metric, experiment_name)

            filename = "results/plots/%s/%s/%s" % (experiment_name, metric, stream_name)
            stream_name_ = "/".join(stream_name.split("/")[0:-1])
            if not os.path.exists("results/plots/%s/%s/%s/" % (experiment_name, metric, stream_name_)):
                os.makedirs("results/plots/%s/%s/%s/" % (experiment_name, metric, stream_name_))

            plt.legend()
            plt.legend(reversed(plt.legend().legendHandles), methods_alias, loc="lower center", ncol=len(methods_alias))
            # plt.title(metric_a+"     "+experiment_name+"     "+stream_name_2)
            plt.ylabel(metric_a)
            # plt.ylim(0, 1)
            plt.xlim(0, len(plot_data)-1)
            plt.xlabel("Data chunk")
            plt.gcf().set_size_inches(10, 5)
            plt.grid(True, color="silver", linestyle=":")
            plt.savefig(filename+".png", bbox_inches='tight')
            plt.savefig(filename+".eps", format='eps', bbox_inches='tight')
            plt.clf()
            plt.close()


def drift_metrics_table_mean(methods, streams, metrics, experiment_names, methods_alias=None, metrics_alias=None, streams_alias=None):

    if methods_alias is None:
        methods_alias = methods
    if metrics_alias is None:
        metrics_alias = metrics

    data = {}
    for experiment_name in experiment_names:
        for clf_name in methods:
            for metric in metrics:
                s_data = []
                for stream_name in streams:
                    try:
                        filename = "results/raw_metrics/%s/%s/%s/%s.csv" % (experiment_name, stream_name, metric, clf_name)
                        if np.isnan(np.mean(np.genfromtxt(filename, delimiter=',', dtype=np.float32))):
                            print("Nan in loading data", stream_name, clf_name, metric, experiment_name)
                        s_data.append(np.mean(np.genfromtxt(filename, delimiter=',', dtype=np.float32)))
                    except Exception:
                        s_data.append(0)
                        print("Error in loading data", stream_name, clf_name, metric, experiment_name)
                data[experiment_name, clf_name, metric, 'mean'] = np.mean(s_data)
                data[experiment_name, clf_name, metric, 'std'] = np.std(s_data)

    best = {}
    for experiment_name in experiment_names:
        for metric in metrics:
            vals = []
            for clf_name in methods:
                vals.append(data[experiment_name, clf_name, metric, 'mean'])

            best[experiment_name, metric] = methods[np.argmin(vals)]
    print(best)


    if not os.path.exists("results/drift_metrics"):
        os.makedirs("results/drift_metrics")
    for metric, metric_a in zip(metrics,metrics_alias):
        table_tex = "\\begin{table}[]\n\\centering\n\\caption{Mean %s on generated %s streams (less is better)}\n" % (metric_a, streams_alias)
        table_tex += "\\scalebox{0.87}{\n\\begin{tabular}{|l|" + "c|"*len(experiment_names) + "}\n"
        table_tex += "\\hline\n & " + " & ".join(experiment_names).upper() + " \\\\ \\hline\n"
        # table_tex = "\\begin{table}[]\n\\centering\n\\caption{$RT_M$ - Mean of recovery time, $RT_S$ - Standard deviation of recovery time, $PL_M$ - Mean of performance loss, $PL_S$ - Standard deviation of performance loss}\n\\begin{tabular}{|l|c|c|c|c|"
        # table_tex += "}\n\\hline\n" + " & $RT_M$ & $RT_S$ & $PL_M$ & $PL_S$" + " \\\\ \\hline\n"
        for method, method_a in zip(methods, methods_alias):
            table_tex += "%s " % method_a
            for experiment_name in experiment_names:
                mean = data[experiment_name, method, metric, 'mean']
                std = data[experiment_name, method, metric, 'std']
                if best[experiment_name, metric] == method:
                    table_tex += "& \\textbf{%0.4f}$\\pm$\\textbf{%0.4f} " % (mean, std)
                else:
                    table_tex += "& %0.4f$\\pm$%0.4f " % (mean, std)
            table_tex += "\\\\ \n"
        table_tex += "\\hline \n"


        table_tex += "\\end{tabular}}\n\\end{table}\n"

        filename = "results/drift_metrics/mean_%s_%s.tex" % (metric, streams_alias)

        print(filename)

        if not os.path.exists("results/drift_metrics/"):
            os.makedirs("results/drift_metrics/")

        with open(filename, "w") as text_file:
            text_file.write(table_tex)
