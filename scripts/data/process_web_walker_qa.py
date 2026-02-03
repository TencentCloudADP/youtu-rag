import json

import pandas as pd
from datasets import load_dataset
from sqlmodel import Session, select

from utu.db.eval_datapoint import DatasetSample
from utu.utils.sqlmodel_utils import SQLModelUtils

engine = SQLModelUtils.get_engine()


def build_dataset():
    # check if exists
    with Session(engine) as session:
        data = session.exec(select(DatasetSample).where(DatasetSample.dataset == "WebWalkerQA")).all()
        if len(data) > 0:
            print("Dataset already exists! Skip.")
            return

    ds = load_dataset("callanwu/WebWalkerQA", split="main")
    data: list[DatasetSample] = []
    for i, row in enumerate(ds.to_list()):
        meta = {"root_url": row["root_url"], **row["info"]}
        level_map = {
            "easy": 1,
            "medium": 2,
            "hard": 3,
        }
        data.append(
            DatasetSample(
                dataset="WebWalkerQA",
                index=i + 1,
                source="WebWalkerQA",
                source_index=i + 1,
                question=row["question"],
                answer=row["answer"],
                topic=row["info"]["domain"],
                level=level_map[row["info"]["difficulty_level"]],
                file_name="",
                meta=meta,
            )
        )
    print(f"Total {len(data)} samples")
    print(f"Sample: {json.dumps(data[0].model_dump(), ensure_ascii=False)}")

    with Session(engine) as session:
        session.add_all(data)
        session.commit()
        print("Upload complete.")


def sampling(dataset_name: str = "WebWalkerQA", n: int = 150):
    assert n % 3 == 0
    import random

    random.seed(42)

    with Session(engine) as session:
        data = session.exec(select(DatasetSample).where(DatasetSample.dataset == dataset_name)).all()
        print(f"Total {len(data)} samples")

    df = pd.DataFrame([d.model_dump() for d in data])
    df_sampled = df.groupby("level").apply(lambda x: x.sample(n=n // 3, random_state=42)).reset_index(drop=True)
    # move `index` to `source_index`; reset index
    df_sampled["source_index"] = df_sampled["index"]
    df_sampled["index"] = range(1, len(df_sampled) + 1)
    df_sampled["dataset"] = f"{dataset_name}_{n}"
    df_sampled.drop(columns=["id"], inplace=True)

    data_sampled = [DatasetSample(**row) for row in df_sampled.to_dict(orient="records")]
    with Session(engine) as session:
        session.add_all(data_sampled)
        session.commit()
        print("Upload complete.")


if __name__ == "__main__":
    build_dataset()
    sampling(n=15)  # randomly sample 15 datapoints for testing
