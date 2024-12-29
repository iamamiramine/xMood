import os
import pickle

from src.application.dataloader.models.dataloader_model import DataloaderModule
from src.domain.constants.paths_constants import DATALOADER_PATH
from src.domain.models.dataloader.dataloader_model import DataloaderParameters


def prepare_dataloader(parameters: DataloaderParameters) -> dict:
    datamodule_parameters = {
        "dataset_name": parameters.dataset_name,
        "context_size": parameters.context_size,
        "max_positions": parameters.max_positions,
        "max_bars": parameters.max_bars,
        "max_bars_per_context": parameters.max_bars_per_context,
        "max_contexts_per_file": parameters.max_contexts_per_file,
        "bar_token_mask": parameters.bar_token_mask,
        "bar_token_idx": parameters.bar_token_idx,
        "batch_size": parameters.batch_size,
        "num_workers": parameters.num_workers,
        "pin_memory": parameters.pin_memory,
        "train_val_test_split": parameters.train_val_test_split,
    }

    dataloader_path = os.path.join(DATALOADER_PATH, parameters.dataset_name)
    if not os.path.exists(dataloader_path):
        os.makedirs(dataloader_path)

    pickle.dump(datamodule_parameters, open(os.path.join(dataloader_path, f"{parameters.dataset_name}_datamodule_parameters.pkl"), "wb"))

    return {"Message": "Dataloader Prepared"}


def run_dataloader(dataset_name: str, load_latent: bool = False, load_desc: bool = False) -> dict:
    datamodule_parameters = pickle.load(open(os.path.join(DATALOADER_PATH, dataset_name, f"{dataset_name}_datamodule_parameters.pkl"), "rb"))

    datamodule_parameters["load_latent"] = load_latent
    datamodule_parameters["load_desc"] = load_desc

    datamodule = DataloaderModule(**datamodule_parameters)
    datamodule.setup()
    datamodule.setup("fit")  # 'fit' loads both train and validation data

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    for batch in train_loader:
        continue
    for batch in val_loader:
        continue
    for batch in test_loader:
        continue
    return {"Message": "Dataloader Run Successfully"}
