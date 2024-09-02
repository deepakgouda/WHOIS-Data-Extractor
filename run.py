from loguru import logger

from src.config import Config
from src.dataset import Dataset
from src.train import Trainer

logger.add("logs/training_{time}.log", enqueue=True)


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

    trainer.run()

    return None


if __name__ == "__main__":
    main()
