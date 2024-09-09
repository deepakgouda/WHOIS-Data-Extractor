import os.path as osp
import torch

from collections import defaultdict
from src.config import Config
from src.dataset import Dataset
from src.train import Trainer

from loguru import logger

logger.add("logs/inference_{time}.log", enqueue=True)


def convert_labels_to_json(text, labels):
    result = defaultdict(list)
    for indx, label in enumerate(labels):
        if label == "O":
            continue
        _, entity = label.split("-")
        result[entity].append(text[indx])
    result = {k: " ".join(v) for k, v in result.items()}
    return result


def main():
    config = Config(logger)

    logger.info("Loading datasets...")
    training_dataset = Dataset(logger=logger).prepare(config=config, partition="train")
    logger.info(f"Training dataset loaded, length={len(training_dataset)}")

    valid_dataset = Dataset(logger=logger).prepare(config=config, partition="valid")
    logger.info(f"Validation dataset loaded, length={len(valid_dataset)}")

    test_dataset = Dataset(logger=logger).prepare(config=config, partition="test")
    logger.info(f"Test dataset loaded, length={len(test_dataset)}")

    trainer = Trainer(
        config=config,
        training_dataset=training_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
    )

    trainer._checkpoint_container = trainer._checkpoint_container.load(
        osp.join(config.model_dir, f"{config.bert_model_name_or_path}.pt")
    )
    trainer._model.load_state_dict(trainer._checkpoint_container.state_dict)

    dataset = trainer._test_dataset
    pred_lbs = trainer.predict(dataset)

    for indx in range(len(dataset)):
        d1 = convert_labels_to_json(dataset.text[indx], dataset.lbs[indx])
        d2 = convert_labels_to_json(dataset.text[indx], pred_lbs[indx])
        if d1 != d2:
            print(indx)

    inference_dataset = Dataset(logger=logger).prepare(
        config=config, partition="inference"
    )
    dataset = inference_dataset
    pred_lbs = trainer.predict(dataset=dataset)

    for indx in range(len(dataset)):
        text = dataset.text[indx]
        output = convert_labels_to_json(dataset.text[indx], pred_lbs[indx])
        input = " ".join(text)
        print(f"INPUT : {input}")
        print(f"OUTPUT : {output}")
        print("---------")


if __name__ == "__main__":
    main()
