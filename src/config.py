import os.path as osp
import json
from dataclasses import dataclass, field
from .utils import entity_to_bio_labels


@dataclass
class Config:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- manage directories and IO ---
    data_dir: str = field(
        default="./data", metadata={"help": "Directory to datasets"}
    )
    model_dir: str = field(
        default="./model", metadata={"help": "Directory to model checkpoints"}
    )
    bert_model_name_or_path: str = field(
        default="distilbert-base-uncased",
        metadata={
            "help": "Path to pretrained BERT model or model identifier from huggingface.co/models; "
            "Used to construct BERT embeddings if not exist"
        },
    )

    # --- training arguments ---
    lr: float = field(default=5e-5, metadata={"help": "learning rate"})
    batch_size: int = field(default=16, metadata={"help": "model training batch size"})
    n_epochs: int = field(
        default=20, metadata={"help": "number of denoising model training epochs"}
    )
    warmup_ratio: float = field(default=0.1, metadata={"help": "ratio of warmup steps"})
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "Learning rate scheduler with warm ups defined in `transformers`, Please refer to "
            "https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#schedules for details",
            "choices": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
        },
    )
    weight_decay: float = field(
        default=0.01, metadata={"help": "strength of weight decay"}
    )
    seed: int = field(default=42, metadata={"help": "random seed"})

    def get_meta(self):
        """
        Load meta file and update arguments
        """

        # Load meta if exist
        meta_file_path = osp.join(self.data_dir, "meta.json")
        assert osp.isfile(meta_file_path), f"Metadata file does not exist!"

        with open(meta_file_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        self.entity_types = meta_dict["entity_types"]
        self.bio_label_types = entity_to_bio_labels(meta_dict["entity_types"])
        return

    def get_config(self):
        """
        Load config file and update arguments
        """

        # Load config if exist
        config_path = osp.join("config.json")
        if not osp.isfile(config_path):
            self._logger.warn(f"Config file does not exist! Using default values.")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        for key, value in config_dict.items():
            setattr(self, key, value)
        return

    def __init__(self, logger):
        self._logger = logger
        self.get_config()
        self.get_meta()

        file_path = osp.join(self.data_dir, "train.json")
        if not osp.isfile(file_path):
            self._logger.error(f"Training file {file_path} does not exist!")
            raise FileNotFoundError(f"Training file {file_path} does not exist!")

