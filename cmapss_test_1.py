import pandas as pd
from pathlib import Path
import os

pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 15)


print("Working directory:", os.getcwd())
print("Files in directory:", os.listdir())

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR/"CMAPSS_Data"/"train_FD001.txt"


columns = (
    ["engine_id", "cycle"] 
    + [f"operational_setting_{i}" for i in range (1,4)] 
    + [f"sensor_measurement_{i}" for i in range (1,22)]
)

data = pd.read_csv(
    file_path,
    sep=r"\s+",
    header=None,
    names=columns
)

print(data.head())
print(data.shape)
print(data.columns)
"""
print("__________________________________________________")
print("__________________________________________________")
print("__________________________________________________")

print(data.head(5))

engine_24 = data[data["engine_id"] == 24]
print(engine_24)

max_cycle_engine_24 = engine_24["cycle"].max()

print(max_cycle_engine_24)


print("______________________________________________________")
print("______________________________________________________")
print("______________________________________________________")

print(engine_24.shape[0])

"""
engine_24 = data[data["engine_id"] == 24]
first_row = engine_24.iloc[0]

documents = []
for _,row in data.iterrows():
    row_text=""
    for column in row.index:
        row_text+=f"{column}: {row[column]} \n"
    documents.append(row_text)
print(documents[0])
print("___________________________________")
result = data[(data["engine_id"] == 24) & (data["cycle"] == 100)]
print(result)

print("_______________________________________________________")
print("Rows in data:", data.shape[0])
print("Number of documents:", len(documents))