import pandas as pd
import pathlib

R = pathlib.Path("data/raw")

df = pd.read_csv(R / "styles.csv", on_bad_lines="skip")   # a few malformed rows exist
df["path"] = df["id"].apply(lambda i: str(R / "images" / f"{i}.jpg"))
df = df[df["path"].apply(lambda p: pathlib.Path(p).exists())]

counts = df["articleType"].value_counts()

df = df[df["articleType"].isin(counts[counts >= 100].index)]   # drop rare classes
df = df.sample(n=min(5500, len(df)), random_state=42).reset_index(drop=True)

cols = ["id", "articleType", "path"]

df.iloc[:500][cols].to_csv("data/queries.csv", index=False)    # held-out queries
df.iloc[500:][cols].to_csv("data/catalog.csv", index=False)    # indexed items
p = df.iloc[500:]["articleType"].value_counts(normalize=True)

print("catalog:", len(df) - 500, "chance P@k:", round((p**2).sum(), 4))

# catalog: 5000 chance P@k: 0.0566