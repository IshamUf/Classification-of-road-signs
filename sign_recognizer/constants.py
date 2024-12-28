import torch

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 20
LEARNING_RATE = 0.0008
BATCH_SIZE = 64

INPUT_DIM = 3 * 50 * 50

OUTPUT_DIM = 43
