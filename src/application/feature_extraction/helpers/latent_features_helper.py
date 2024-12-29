import glob
import os
import pickle

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import plotly.express as px
from pyntcloud import PyntCloud

import skfuzzy as fuzz
from transformers.models.bert.modeling_bert import BertAttention

from test_scripts.feature_extraction.ann_model import Model1
from src.domain.constants.paths_constants import LATENTS_PATH, FEATURES_PATH, ANN_PATH, LABELS_PATH, MIDI_PATH
from src.application.feature_extraction.models.vae_model import VqVaeModule


def load_vae_from_checkpoint(checkpoint_dir: str):
    pl_ckpt = torch.load(checkpoint_dir, map_location="cpu")
    kwargs = pl_ckpt["hyper_parameters"]
    model = VqVaeModule(**kwargs)
    state_dict = pl_ckpt["state_dict"]
    # position_ids are no longer saved in the state_dict starting with transformers==4.31.0
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith("embeddings.position_ids")}
    try:
        # succeeds for checkpoints trained with transformers>4.13.0
        model.load_state_dict(state_dict)
    except RuntimeError:
        # work around a breaking change introduced in transformers==4.13.0, which fixed the position_embedding_type of cross-attention modules "absolute"
        config = model.transformer.decoder.bert.config
        for layer in model.transformer.decoder.bert.encoder.layer:
            layer.crossattention = BertAttention(config, position_embedding_type=config.position_embedding_type)
        model.load_state_dict(state_dict)
    model.freeze()
    model.eval()
    model.cpu()
    return model


def read_labels(dataset_name, split: str = None):
    path = os.path.join(LABELS_PATH, dataset_name, split, f"{split}_labels.csv") if split else os.path.join(LABELS_PATH, dataset_name, "labels.csv")
    labels = pd.read_csv(str(path))
    label_columns = labels.columns
    label_columns = label_columns.drop("file")
    midi_files = glob.glob(os.path.join(os.path.join(MIDI_PATH, dataset_name), "**/*.mid"), recursive=True)

    all_labels = []

    for i in midi_files:
        row = labels.loc[labels["file"] == (os.path.basename(i))]
        if not len(row):
            continue

        row_labels = {"file_name": str(os.path.basename(i))}
        for label_column in label_columns:
            row_labels[label_column] = row[label_column].values[0]
        all_labels.append(row_labels)

    all_labels = pd.DataFrame(all_labels)  # labels
    return all_labels, label_columns


def read_label_for_midi(dataset_name, midi_file_path, split: str = None):
    # Construct the path to the CSV label file
    path = os.path.join(LABELS_PATH, dataset_name, split, f"{split}_labels.csv") if split else os.path.join(LABELS_PATH, dataset_name, "labels.csv")
    labels = pd.read_csv(str(path))

    # Drop "file" from columns to isolate label columns
    label_columns = labels.columns.drop("file")

    # Extract only the basename of the MIDI file to match the CSV file format
    midi_file_name = os.path.basename(midi_file_path)

    # Locate the row corresponding to the specific MIDI file
    row = labels.loc[labels["file"] == midi_file_name]
    if not len(row):
        print(f"No labels found for {midi_file_name}")
        return None  # Return None if no label is found for this MIDI file

    # Prepare a dictionary for the MIDI file's labels
    midi_labels = {"file_name": midi_file_name}
    for label_column in label_columns:
        midi_labels[label_column] = row[label_column].values[0]

    # Convert to DataFrame for consistency with the original function's return format
    midi_labels_df = pd.DataFrame([midi_labels])
    return midi_labels_df, label_columns


def read_features(dataset_name):
    midi_files = glob.glob(os.path.join(os.path.join(MIDI_PATH, dataset_name), "**/*.mid"), recursive=True)

    all_latents_individual = []

    for i in midi_files:
        latent_file = os.path.join(
            str(LATENTS_PATH),
            dataset_name,
            str(os.path.basename(i)),
        )

        sample = pickle.load(open(latent_file, "rb"))

        features = {}
        for j, feature in enumerate(sample["latents"].cpu().numpy()[0]):
            features[f"feature_{j}"] = feature
        features["file_name"] = str(os.path.basename(i))
        all_latents_individual.append(features)

    all_latents_individual = pd.DataFrame(all_latents_individual)  # labels
    all_latents_individual.fillna(0, inplace=True)

    return all_latents_individual


def cluster_reduced_data(latent_vectors, labels):
    data = pd.DataFrame(latent_vectors)
    data_features = data.shape[1]
    reduced_columns = []
    for i in range(data_features):
        reduced_columns.append(f"feature_{i}")
    data.columns = reduced_columns

    data = pd.concat([data, labels], axis=1)
    alldata = np.vstack(([data[column] for column in data.columns]))

    fpcs = []
    plt.figure(figsize=(8, 8))
    c = labels.shape[1]
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, c, 2, error=0.005, maxiter=1000, init=None)

    # Store fpc values for later
    fpcs.append(fpc)
    group = []
    cluster_membership = np.argmax(u, axis=0)

    for i in cluster_membership:
        group.append(i)

    data["group"] = group

    label_vectors = {}
    for i in range(c):
        label_vectors[i] = {}

    for i in range(0, c):
        for column in data.columns:
            if column != "group":
                label_vectors[i][column] = data[data["group"] == i][column].mean()

    return data, label_vectors, cntr


def plot_reduced_data(data, cntr):
    fig = px.scatter(data, x="X", y="Y", color="group", hover_name="file_name", hover_data=["anger", "love", "joy", "sadness", "surprise"])
    for pt in cntr:
        fig.add_scatter(
            dx=pt[0],
            dy=pt[1],
            marker=dict(color="red", size=200),
            mode="markers",
        )
    fig.show()


def export_features_to_csv(all_latents_individual: pd.DataFrame, dataset_name: str):
    path = os.path.join(FEATURES_PATH, dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    all_latents_individual.to_csv(os.path.join(path, "features.csv"), index=False)


def conv_to_ply(args, vis_data):
    data = pd.DataFrame(vis_data)
    data.columns = ["X", "Y", "Z"]
    data["R"] = 1.0
    data["G"] = 1.0
    data["B"] = 1.0
    # alldata = np.vstack((data["X"], data["Y"]))

    d = {"x": data["X"], "y": data["Y"], "z": data["Z"], "red": data["R"], "green": data["G"], "blue": data["B"]}

    cloud = PyntCloud(pd.DataFrame(data=d))
    if not os.path.exists((os.path.join(str(LATENTS_PATH), args.dataset_name, "ply"))):
        os.makedirs((os.path.join(str(LATENTS_PATH), args.dataset_name, "ply")))
    cloud.to_file(os.path.join(str(LATENTS_PATH), args.dataset_name, "ply", f"{args.dataset_name}_{args.n_codes}_point_cloud.ply"))


def rank_correlation(correlation_matrix):
    for label in correlation_matrix:
        correlation_matrix.loc[:, label] = correlation_matrix.loc[:, label].apply(lambda x: abs(x))
    rank = {label: [] for label in correlation_matrix.columns}
    for label in correlation_matrix.columns:
        l = list(enumerate(correlation_matrix[label]))
        l.sort(key=lambda x: x[1], reverse=True)
        rank[label] = l
    return correlation_matrix.rank(ascending=False, axis=1)


def plot_correlation(correlation_matrix):
    import seaborn as sb

    dataplot = sb.heatmap(correlation_matrix, cmap="YlGnBu", annot=True)
    # f = plt.figure(figsize=(19, 15))
    # f.set_figwidth(5)
    # f.set_figheight(5)
    # plt.matshow(correlation_matrix, fignum=1)
    # plt.xticks(range(correlation_matrix.columns.values.shape[0]), correlation_matrix.columns.values, fontsize=12, rotation=45)
    # plt.yticks(range(correlation_matrix.index.values.shape[0]), correlation_matrix.index.values, fontsize=12)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=14)
    # plt.title("Correlation Matrix", fontsize=16)
    plt.show()


def model_using_ann(args, X_train, X_test, y_train, y_test, y_test_file_names, y_labels):
    model1 = Model1(X_train.shape[1], y_train.shape[1])
    model1.fit(x=X_train, y=y_train, epochs=100)
    pred = model1.predict(X_test)

    from src.application.sentiment_learner.services.sentiment_learner_service import calculate_and_print_metrics

    predictions_path = str(os.path.join(ANN_PATH, args.dataset_name))
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    mse, rmse, mae, summ = calculate_and_print_metrics(y_test.to_numpy()[:, :].astype(float), pred)
    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, PCC: {summ}")

    pred = pd.DataFrame(pred, columns=y_labels)
    pred["file_name"] = y_test_file_names.values

    pred.to_csv(str(os.path.join(predictions_path, f"predictions_{args.n_codes}.csv")), index=False)

    return mse, rmse, mae, summ
