import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Commands for GTSRB project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # ------------------- train subcommand -------------------
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument(
        "--data-path", type=str, required=True, help="Path to GTSRB dataset."
    )
    train_parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs."
    )
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")

    # ------------------- test subcommand --------------------
    test_parser = subparsers.add_parser("test", help="Run testing")
    test_parser.add_argument(
        "--data-path", type=str, required=True, help="Path to GTSRB dataset."
    )
    test_parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt).",
    )
    test_parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")

    # ------------------- infer subcommand -------------------
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt).",
    )
    infer_parser.add_argument(
        "--image-path", type=str, required=True, help="Path to the input image."
    )

    args = parser.parse_args()

    if args.command == "train":
        cmd = [
            sys.executable,
            os.path.join("sign_recognizer", "train.py"),
            "--data-path",
            args.data_path,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
        ]
        subprocess.run(cmd, check=True)

    elif args.command == "test":
        cmd = [
            sys.executable,
            os.path.join("sign_recognizer", "test.py"),
            "--data-path",
            args.data_path,
            "--ckpt-path",
            args.ckpt_path,
            "--batch-size",
            str(args.batch_size),
        ]
        subprocess.run(cmd, check=True)

    elif args.command == "infer":
        cmd = [
            sys.executable,
            os.path.join("sign_recognizer", "infer.py"),
            "--ckpt-path",
            args.ckpt_path,
            "--image-path",
            args.image_path,
        ]
        subprocess.run(cmd, check=True)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
