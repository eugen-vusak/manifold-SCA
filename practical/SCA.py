import csv
from collections import OrderedDict
from pathlib import Path

import numpy as np
import umap
from sklearn import manifold, random_projection
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from SCA_measure import compute_guessing_entropy
from utils.parameters import generate_params_grid
from utils.transformers import DummyTranformer

# directories
data_root = Path("data")
results_root = Path("results")
report_filename = Path("report.csv")
ge_se_report_filename = Path("ge_se_report.csv")


traces_filename = "traces_50.csv"
labels_filename = "model.csv"
key_guesses_filename = "key_guesses_ALL.csv"
secret_key_filename = "secret_key.csv"

# datasets
datasets = [
    # "DPAv4",
    # "AES Shivam",
    "Random Delay"
]

models = [
    # "HW",
    "value"
]

number_of_traces_for_dataset = {
    "DPAv4": 250,
    "AES Shivam": 25000,
    "Random Delay": 25000
}

# train test split
train_size = 10000
test_size = 25000

# manifold transform
mf_parameters_dict = OrderedDict({
    "n_components": OrderedDict({
        # 2: {"n_neighbors": [10]},
        # 3: {"n_neighbors": [10]},
        # 5: {"n_neighbors": [10, 20, 30]},
        # 7: {"n_neighbors": [10, 20, 30, 40]},
        # 10: {"n_neighbors": [10, 20, 30, 50, 70]},
        # 15: {"n_neighbors": [10, 30, 50, 70, 100, 150]},
        # 20: {"n_neighbors": [10, 50, 100, 150, 200, 250]},
        # 25: {"n_neighbors": [10, 50, 100, 200, 300, 400]},
        # 30: {"n_neighbors": [10, 50, 100, 300, 500]},
        # 40: {"n_neighbors": [50, 200, 500, 1000]}

        # None: {"n_neighbors": [None]},

        # # 2: {"n_neighbors": [None]},
        # 3: {"n_neighbors": [None]},
        # # 5: {"n_neighbors": [None]},
        # 7: {"n_neighbors": [None]},
        # 10: {"n_neighbors": [None]},
        # 15: {"n_neighbors": [None]},
        # # 20: {"n_neighbors": [None]},
        # 25: {"n_neighbors": [None]},
        # # 30: {"n_neighbors": [None]},
        # # 40: {"n_neighbors": [None]}
        
        3: {"n_neighbors": [10]},
        7: {"n_neighbors": [10, 20, 40]},
        10: {"n_neighbors": [10, 30, 70]},
        15: {"n_neighbors": [10, 50, 70, 150]},
        25: {"n_neighbors": [10, 50, 400]},
        40: {"n_neighbors": [50, 500, 1000]}
    }),
})

mf_n_jobs = None

# clasification
rf_n_jobs = None
gs_n_jobs = None
gs_parameters = {'n_estimators': [100]}


def generate_tranformations(n_neighbors, n_components):

    # dummy_clf = DummyTranformer()

    # isomap_clf = manifold.Isomap(n_neighbors, n_components=n_components,
    #                              n_jobs=mf_n_jobs)

    lle_clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=n_components,
                                              eigen_solver='auto',
                                              method='standard',
                                              n_jobs=mf_n_jobs)

    mlle_clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=n_components,
                                               method='modified',
                                               n_jobs=mf_n_jobs)

    hlle_clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=n_components,
                                               eigen_solver='dense',
                                               method='hessian',
                                               n_jobs=mf_n_jobs)

    ltsa_clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=n_components,
                                               eigen_solver='dense',
                                               method='ltsa',
                                               n_jobs=mf_n_jobs)

    # umap_clf = umap.UMAP(n_neighbors, n_components=n_components)

    # pca_clf = PCA(n_components=n_components, random_state=1)

    # grp_clf = random_projection.GaussianRandomProjection(n_components=n_components)

    # srp_clf = random_projection.SparseRandomProjection(n_components=n_components)

    # manifold transformations
    trans_clfs = [
        # (dummy_clf, "dummy"),
        # (isomap_clf, "isomap"),
        (lle_clf, "lle"),
        (mlle_clf, "mlle"),
        (hlle_clf, "hlle"),
        (ltsa_clf, "ltsa"),
        # (umap_clf, "umap"),
        # (pca_clf, "pca"),
        # (grp_clf, "grp"),
        # (srp_clf, "srp"),
    ]

    return trans_clfs


def generate_model(trans_clf):

    rp_clf = random_projection.SparseRandomProjection(n_components=30)

    # clasificatior
    rf_clf = RandomForestClassifier(
        n_estimators=100, random_state=1, n_jobs=rf_n_jobs)

    # model pipeline
    model = Pipeline([
        ("random projection", rp_clf),
        ("manifold trnasform", trans_clf),
        ("random_forest", rf_clf)
    ])

    return model


def run_expiriment(X, y,
                   train_size, test_size,
                   trans_clfs,
                   key_guesses, secret_key, n_traces):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, train_size=train_size, shuffle=False)

    for trans_clf, trans_short_name in trans_clfs:

        print("Start {}".format(trans_short_name))

        try:

            # training
            clf_model = generate_model(trans_clf)
            clf_model.fit(X_train, y_train)

            # inference
            y_pred = clf_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            #probabilities, GE and SR
            y_pred_proba = clf_model.predict_proba(X_test)

            guessing_entropy, success_rate = compute_guessing_entropy(
                y_pred_proba,
                key_guesses, secret_key,
                n_traces
            )

            yield acc, guessing_entropy, success_rate

        except Exception as e:
            print("Failed {}".format(trans_short_name))
            print(e)
            yield -1, [], []


def load_data(dataset, model):
    data_folder = data_root/dataset/model

    traces = np.loadtxt(data_folder/traces_filename)
    labels = np.loadtxt(data_folder/labels_filename)
    
    print(traces.shape)
    exit()

    secret_key = np.loadtxt(
        data_root/dataset/secret_key_filename, dtype=np.int)
    key_guesses = np.loadtxt(
        data_folder/key_guesses_filename, dtype=np.int)

    return traces, labels, key_guesses, secret_key


def main():
    with open(report_filename, "a") as report_file, \
            open(ge_se_report_filename, "a") as ge_se_report_file:

        report_writer = csv.writer(report_file)
        gs_se_report_writer = csv.writer(ge_se_report_file)

        for dataset in datasets:
            for model in models:

                print("Running dataset {} in {} leakage model".format(
                    dataset, model))

                traces, labels, key_guesses, secret_key = load_data(
                    dataset, model)

                n_traces = number_of_traces_for_dataset[dataset]

                for params in generate_params_grid(mf_parameters_dict):

                    print("params:", params)

                    trans_clfs = generate_tranformations(n_components=params["n_components"],
                                                         n_neighbors=params["n_neighbors"])

                    try:
                        rezult = run_expiriment(traces, labels,
                                                train_size, test_size,
                                                trans_clfs,
                                                key_guesses, secret_key, n_traces)

                        # write report
                        for (_, trans_short_name), (acc, ge, se) in zip(trans_clfs, rezult):
                            trans_short_name = "srp30+" + trans_short_name

                            print("Report:")
                            print("\tDataset:   {}".format(dataset))
                            print("\tModel:     {}".format(model))
                            print("\tManifold:  {}".format(
                                trans_short_name))
                            print("\tAccuracy:  {}".format(acc))
                            print()

                            # write general report
                            report_writer.writerow([dataset, model,
                                                    train_size, test_size,
                                                    params["n_components"],
                                                    params["n_neighbors"],
                                                    trans_short_name, acc])

                            # write GE report
                            gs_se_report_writer.writerow([dataset, model,
                                                          train_size, test_size,
                                                          params["n_components"],
                                                          params["n_neighbors"],
                                                          trans_short_name,
                                                          "GE", *ge])

                            # write SE report
                            gs_se_report_writer.writerow([dataset, model,
                                                          train_size, test_size,
                                                          params["n_components"],
                                                          params["n_neighbors"],
                                                          trans_short_name,
                                                          "SE", *se])

                    except Exception as e:
                        print("Experiment ", "_".join(map(str,
                                                          [dataset, model, train_size, test_size])), "failed")
                        print(e)

                    report_file.flush()
                    ge_se_report_file.flush()


if __name__ == "__main__":
    main()
