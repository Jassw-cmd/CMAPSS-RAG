from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "CMAPSS_Data" / "train_FD001.txt"

columns =(
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range (1,4)]
    + [f"sensor_{i}" for i in range (1, 22)]
)

def load_cmapss():
    data = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=columns
    )
    return data

if __name__ == "__main__":
    data = load_cmapss()
    print("Shape:", data.shape)
    print(data.head())
    print(data.columns.tolist())