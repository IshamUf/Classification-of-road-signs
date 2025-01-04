import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from sign_recognizer.dataset import GTSRB
from sign_recognizer.model import GtsrbModel
from sign_recognizer.trainer import LitGTSRBTrainer
from sign_recognizer.utils.compose_builder import config_compose


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Основная функция для тренировки модели.

    Шаги:
        1. Парсинг аргументов из конфигов.
        2. Создание тренировочного и валидационного датасетов.
        3. Настройка трансформаций для данных.
        4. Инициализация модели и тренера.
        5. Запуск процесса обучения.
        6. Сохранение чекпоинта модели.
    """

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data_cfg = cfg.data
    model_cfg = cfg.model
    trainer_cfg = cfg.trainer
    transforms_cfg = cfg.transforms

    dataset = GTSRB(root=data_cfg.root_dir, split="train", transform=None)

    val_ratio = trainer_cfg.val_ratio
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_transforms = config_compose(transforms_cfg.train_transforms)
    val_transforms = config_compose(transforms_cfg.val_transforms)

    train_ds.dataset.transform = train_transforms
    val_ds.dataset.transform = val_transforms

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
    )

    base_model = GtsrbModel(output_dim=model_cfg.output_dim)

    lit_model = LitGTSRBTrainer(
        model=base_model, learning_rate=trainer_cfg.learning_rate
    )

    trainer = pl.Trainer(
        max_epochs=trainer_cfg.epochs,
        accelerator="auto",
        devices="auto",
    )

    trainer.fit(lit_model, train_loader, val_loader)

    ckpt_path = trainer_cfg.ckpt_path
    trainer.save_checkpoint(ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
