import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from sign_recognizer.dataset import GTSRB
from sign_recognizer.model import GtsrbModel
from sign_recognizer.trainer import LitGTSRBTrainer
from sign_recognizer.utils.compose_builder import config_compose


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Основная функция для тестирования модели на тестовом датасете.

    Шаги:
        1. Парсинг аргументов из конфигов.
        2. Создание тестового датасета и загрузчика данных.
        3. Загрузка модели из указанного чекпоинта.
        4. Выполнение тестирования через PyTorch Lightning Trainer.
    """
    data_cfg = cfg.data
    model_cfg = cfg.model
    trainer_cfg = cfg.trainer
    transforms_cfg = cfg.transforms

    test_dataset = GTSRB(root=data_cfg.root_dir, split="test", transform=None)
    test_transform = config_compose(transforms_cfg.val_transforms)
    test_dataset.transform = test_transform

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
    )

    base_model = GtsrbModel(output_dim=model_cfg.output_dim)
    ckpt_path = trainer_cfg.ckpt_path
    lit_model = LitGTSRBTrainer.load_from_checkpoint(
        checkpoint_path=ckpt_path, model=base_model
    )
    trainer = pl.Trainer(accelerator="auto", devices="auto")
    trainer.test(lit_model, test_loader)


if __name__ == "__main__":
    main()
