import argparse
import subprocess
import sys

from dvc.repo import Repo


def get_data():
    with Repo() as repo:
        repo.pull()


def main():
    parser = argparse.ArgumentParser(
        description="Команды для commands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Запуск обучения")
    train_parser.add_argument(
        "overrides",
        nargs="*",
        help="Дополнительные переопределения для Hydra (например: trainer.epochs=10)",
    )

    test_parser = subparsers.add_parser("test", help="Запуск тестирования")
    test_parser.add_argument(
        "overrides",
        nargs="*",
        help="Дополнительные переопределения для Hydra (например: data.batch_size=32)",
    )

    infer_parser = subparsers.add_parser("infer", help="Запуск инференса")
    infer_parser.add_argument(
        "overrides",
        nargs="*",
        help="Переопределения для Hydra (например: infer.image_path=some_image.png)",
    )

    args = parser.parse_args()

    if args.command == "train":
        get_data()
        cmd = [
            sys.executable,
            "sign_recognizer/train.py",
        ] + args.overrides
        subprocess.run(cmd, check=True)

    elif args.command == "test":
        get_data()
        cmd = [
            sys.executable,
            "sign_recognizer/test.py",
        ] + args.overrides
        subprocess.run(cmd, check=True)

    elif args.command == "infer":
        cmd = [
            sys.executable,
            "sign_recognizer/infer.py",
        ] + args.overrides
        subprocess.run(cmd, check=True)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
