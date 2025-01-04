import hydra
import torch
from omegaconf import DictConfig
from PIL import Image

from sign_recognizer.model import GtsrbModel
from sign_recognizer.trainer import LitGTSRBTrainer
from sign_recognizer.utils.compose_builder import config_compose


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Основной скрипт для выполнения инференса на модели GTSRB.
    Шаги:
        1. Парсит аргументы из конфигов.
        2. Загружает модель из указанного чекпоинта.
        3. Применяет предобработку к входной картинке.
        4. Выполняет предсказание класса для входной картинки.
        5. Выводит предсказанный класс в консоль.
    """
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_cfg = cfg.model
    trainer_cfg = cfg.trainer
    transforms_cfg = cfg.transforms

    infer_transform = config_compose(transforms_cfg.infer_transforms)

    image_path = cfg.get("infer", {}).get("image_path", None)

    base_model = GtsrbModel(output_dim=model_cfg.output_dim)
    ckpt_path = trainer_cfg.ckpt_path
    lit_model = LitGTSRBTrainer.load_from_checkpoint(
        checkpoint_path=ckpt_path, model=base_model
    )
    lit_model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = infer_transform(image).unsqueeze(0)  # (1, C, H, W)

    with torch.no_grad():
        logits = lit_model(image_tensor)
        pred = torch.argmax(logits, dim=1)
        class_id = pred.item()

    print(f"Предсказанный класс: {class_id}")


if __name__ == "__main__":
    main()
