from core import calculate_metrics
from core import plot_streams_matplotlib
from core import drift_metrics_table_mean
from core import pairs_metrics_multi
from core import plot_streams_mean
from core import plot_radars

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

stream_sets = []
streams_aliases = []

streams = []

directory = "cluster_setup/"
mypath = "results/raw_conf/cs/%s" % directory
streams += ["%s%s" % (directory, f) for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]

stream_sets += [streams]
streams_aliases += ["method"]


method_names = [
               "OCWE-AC",
               "OCWE-BC",
               "OCWE-SC",
               "OCWE-KM",
               "OCWE-MB",
                ]

methods_alias = [
               "Aglom",
               "Birch",
               "Spect",
               "KMeans",
               "MBKM",
                ]


metrics_alias = [
           "Gmean",
           "F-score",
           "Precision",
           "Recall",
           "Specificity",
          ]

metrics = [
           "g_mean",
           "f1_score",
           "precision",
           "recall",
           "specificity",
          ]


experiment_names = [
                    "cs"
                    ]

for streams, streams_alias in zip(stream_sets, streams_aliases):
    for experiment_name in experiment_names:
        calculate_metrics(method_names, streams, metrics, experiment_name, recount=True)
        # plot_streams_matplotlib(method_names, streams, metrics, experiment_name, gauss=2, methods_alias=methods_alias, metrics_alias=metrics_alias)

    # pairs_metrics_multi(method_names, streams, metrics, experiment_names, methods_alias=methods_alias, metrics_alias=metrics_alias, streams_alias=streams_alias, title=True)
    plot_radars(method_names, streams, metrics, experiment_name, metrics_alias=metrics_alias, methods_alias=methods_alias)