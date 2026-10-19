"""Load MovieLens 1M, verify it, and write the time-based split.

    python -m src.data   ->  data/processed/{ratings,movies,users}.parquet
"""
import hashlib
import re

import pandas as pd

from src.config import PROCESSED, RAW, TEST_QUANTILE, VAL_QUANTILE, ZIP_MD5


def verify_zip() -> bool:
    z = RAW.parent / "ml-1m.zip"
    if not z.exists():
        print("ml-1m.zip not found next to data/raw/ml-1m (checksum skipped)")
        return False
    ok = hashlib.md5(z.read_bytes()).hexdigest() == ZIP_MD5
    print("ml-1m.zip MD5", "matches GroupLens" if ok else "DOES NOT MATCH GroupLens")
    return ok


def _read(name: str, cols: list[str]) -> pd.DataFrame:
    return pd.read_csv(RAW / name, sep="::", engine="python", names=cols, encoding="latin-1")


def load_raw():
    ratings = _read("ratings.dat", ["user_id", "item_id", "rating", "ts"])
    movies = _read("movies.dat", ["item_id", "title", "genres"])
    users = _read("users.dat", ["user_id", "gender", "age", "occupation", "zip"])
    year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(int)
    movies["year"] = year
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    # "Matrix, The" -> "The Matrix" for display
    movies["clean_title"] = movies["clean_title"].map(
        lambda t: re.sub(r"^(.*), (The|A|An|Les|La|Le|L'|Il|Das|Die|El)$", r"\2 \1", t))
    movies["genres"] = movies["genres"].str.split("|")
    return ratings, movies, users


def split(ratings: pd.DataFrame) -> pd.DataFrame:
    t_val = ratings["ts"].quantile(VAL_QUANTILE)
    t_test = ratings["ts"].quantile(TEST_QUANTILE)
    ratings = ratings.copy()
    ratings["split"] = "train"
    ratings.loc[ratings["ts"] >= t_val, "split"] = "val"
    ratings.loc[ratings["ts"] >= t_test, "split"] = "test"
    return ratings


def main():
    verify_zip()
    ratings, movies, users = load_raw()
    ratings = split(ratings)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(PROCESSED / "ratings.parquet", index=False)
    movies.to_parquet(PROCESSED / "movies.parquet", index=False)
    users.to_parquet(PROCESSED / "users.parquet", index=False)
    for s in ["train", "val", "test"]:
        part = ratings[ratings["split"] == s]
        d = pd.to_datetime(part["ts"], unit="s")
        print(f"{s:5s} {len(part):>8,} ratings  {part.user_id.nunique():>5,} users  "
              f"{d.min():%Y-%m-%d} -> {d.max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
