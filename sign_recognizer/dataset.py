import csv
import pathlib
from typing import Any, Callable, Optional, Tuple

import PIL
from torch.utils.data import Dataset


class GTSRB(Dataset):
    """
    Класс датасета GTSRB (German Traffic Sign Recognition Benchmark).
    Аргументы:
        root: путь к папке с файлами (Train.csv / Test.csv, а также картинками).
        split: 'train' или 'test' — по нему определяем, какой CSV-файл читать.
        transform: функция или compose-трансформация,
                   которая будет применяться к каждой картинке.
    """

    def __init__(self, root: str, split: str, transform: Optional[Callable] = None):
        self.base_folder = pathlib.Path(root)
        self.csv_file = self.base_folder / (
            "Train.csv" if split == "train" else "Test.csv"
        )

        with open(str(self.csv_file)) as csvfile:
            samples = [
                (str(self.base_folder / row["Path"]), int(row["ClassId"]))
                for row in csv.DictReader(csvfile, delimiter=",", skipinitialspace=True)
            ]

        self.samples = samples
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        image_path, class_id = self.samples[index]
        image = PIL.Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, class_id
