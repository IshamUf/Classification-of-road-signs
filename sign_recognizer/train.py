import argparse

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from sign_recognizer.constants import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT_DIM,
    SEED,
)
from sign_recognizer.dataset import GTSRB
from sign_recognizer.model import GtsrbModel
from sign_recognizer.trainer import LitGTSRBTrainer


def parse_args():
    """
    Парсинг аргументов для тренировки модели.

    Возвращает:
        argparse.Namespace: Аргументы, включающие:
            - data-path: Путь до датасета.
            - epochs: Количество эпох.
            - batch-size: Размер батча.
            - learning-rate: Скорость обучения.
            - ckpt-path: Путь для сохранения модели.
    """
    parser = argparse.ArgumentParser(description="Train script.")
    parser.add_argument(
        "--data-path", type=str, required=True, help="Путь до датасета."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Кол-во эпох.")
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Размер бача."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate."
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="Models/gtsrb_model.ckpt",
        help="Путь для сохранения.",
    )
    return parser.parse_args()


def main():
    """
    Основная функция для тренировки модели.

    Шаги:
        1. Парсинг аргументов командной строки.
        2. Создание тренировочного и валидационного датасетов.
        3. Настройка трансформаций для данных.
        4. Инициализация модели и тренера.
        5. Запуск процесса обучения.
        6. Сохранение чекпоинта модели.
    """
    args = parse_args()
    dataset = GTSRB(root=str(args.data_path), split="train", transform=None)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    torch.manual_seed(SEED)
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    train_transforms = T.Compose(
        [
            T.ColorJitter(brightness=1.0, contrast=0.5, saturation=1, hue=0.1),
            T.RandomEqualize(p=0.4),
            T.AugMix(),
            T.RandomHorizontalFlip(p=0.3),
            T.RandomVerticalFlip(p=0.3),
            T.GaussianBlur(kernel_size=(3, 3)),
            T.RandomRotation(30),
            T.Resize([50, 50]),
            T.ToTensor(),
        ]
    )

    val_transforms = T.Compose([T.Resize([50, 50]), T.ToTensor()])

    train_subset.dataset.transform = train_transforms
    val_subset.dataset.transform = val_transforms

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True
    )

    base_model = GtsrbModel(output_dim=OUTPUT_DIM)
    model = LitGTSRBTrainer(model=base_model, learning_rate=LEARNING_RATE)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.save_checkpoint(args.ckpt_path)
    print(f"Model checkpoint saved to {args.ckpt_path}")


if __name__ == "__main__":
    main()
