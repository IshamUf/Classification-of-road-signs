import argparse

import torch
import torchvision.transforms as T
from PIL import Image

from sign_recognizer.constants import OUTPUT_DIM
from sign_recognizer.model import GtsrbModel
from sign_recognizer.trainer import LitGTSRBTrainer


def parse_args():
    """
    Парсинг аргументов для инференса.

    Возвращает:
        argparse.Namespace: Объект, содержащий аргументы:
            - `ckpt_path`: Путь к файлу чекпоинта модели (.ckpt).
            - `image_path`: Путь к входной картинке для инференса.
    """
    parser = argparse.ArgumentParser(description="Infer script.")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Путь к чекпоинту модели (.ckpt).",
    )
    parser.add_argument(
        "--image-path", type=str, required=True, help="Путь до картинки."
    )
    return parser.parse_args()


def main():
    """
    Основной скрипт для выполнения инференса на модели GTSRB.
    Шаги:
        1. Парсит аргументы командной строки.
        2. Загружает модель из указанного чекпоинта.
        3. Применяет предобработку к входной картинке.
        4. Выполняет предсказание класса для входной картинки.
        5. Выводит предсказанный класс в консоль.
    """
    args = parse_args()
    base_model = GtsrbModel(output_dim=OUTPUT_DIM)
    lit_model = LitGTSRBTrainer.load_from_checkpoint(
        checkpoint_path=args.ckpt_path, model=base_model
    )
    lit_model.eval()
    infer_transform = T.Compose([T.Resize([50, 50]), T.ToTensor()])
    image = Image.open(args.image_path).convert("RGB")
    image_tensor = infer_transform(image).unsqueeze(0)  # (1, C, H, W)
    with torch.no_grad():
        logits = lit_model(image_tensor)
        pred = torch.argmax(logits, dim=1)
        class_id = pred.item()
    print(f"Предсказанный класс: {class_id}")


if __name__ == "__main__":
    main()
