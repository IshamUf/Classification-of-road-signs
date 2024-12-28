import argparse

import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader

from sign_recognizer.constants import OUTPUT_DIM
from sign_recognizer.dataset import GTSRB
from sign_recognizer.model import GtsrbModel
from sign_recognizer.trainer import LitGTSRBTrainer


def parse_args():
    """
    Парсинг аргументов для тестирования.

    Возвращает:
        argparse.Namespace: Аргументы, включающие:
            - data-path: Путь до тестового датасета.
            - ckpt-path: Путь до чекпоинта модели.
            - batch-size: Размер батча.
    """
    parser = argparse.ArgumentParser(description="Test script.")
    parser.add_argument(
        "--data-path", type=str, required=True, help="Путь до датасета."
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Путь к чекпоинту модели (.ckpt).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Размер бача.")
    return parser.parse_args()


def main():
    """
    Основная функция для тестирования модели на тестовом датасете.

    Шаги:
        1. Парсинг аргументов командной строки.
        2. Создание тестового датасета и загрузчика данных.
        3. Загрузка модели из указанного чекпоинта.
        4. Выполнение тестирования через PyTorch Lightning Trainer.
    """
    args = parse_args()
    test_transform = T.Compose([T.Resize([50, 50]), T.ToTensor()])
    test_dataset = GTSRB(root=args.data_path, split="test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    base_model = GtsrbModel(output_dim=OUTPUT_DIM)
    lit_model = LitGTSRBTrainer.load_from_checkpoint(
        checkpoint_path=args.ckpt_path, model=base_model
    )
    trainer = pl.Trainer(accelerator="auto", devices="auto")
    trainer.test(lit_model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
