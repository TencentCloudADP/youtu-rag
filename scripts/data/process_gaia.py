import json

import datasets
from huggingface_hub import login, snapshot_download
from sqlmodel import Session, select

from utu.db.eval_datapoint import DatasetSample
from utu.utils import DIR_ROOT, EnvUtils, SQLModelUtils

login(EnvUtils.get_env("HF_TOKEN"))
engine = SQLModelUtils.get_engine()


def build_dataset(split: str) -> datasets.Dataset:
    dataset_id = f"GAIA_{split}"
    data_dir = DIR_ROOT / "data" / "gaia"

    # check if exists
    with Session(engine) as session:
        data = session.exec(select(DatasetSample).where(DatasetSample.dataset == dataset_id)).all()
        if len(data) > 0:
            print("Dataset already exists! Skip.")
            return

    if not data_dir.exists():
        snapshot_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
            local_dir=str(data_dir),
            ignore_patterns=[".gitattributes", "README.md"],
        )

    ds = datasets.load_dataset(
        str(data_dir / "GAIA.py"),
        name="2023_all",
        split=split,
        trust_remote_code=True,
    )

    data: list[DatasetSample] = []
    for i, row in enumerate(ds.to_list()):
        data.append(
            DatasetSample(
                dataset=dataset_id,
                index=i + 1,
                source="GAIA",
                source_index=i + 1,
                question=row["Question"],
                answer=row["Final answer"],
                topic="",
                level=row["Level"],
                file_name=row["file_name"],
                meta={key: row.get(key, None) for key in ["task_id", "Annotator Metadata"]},
            )
        )
    print(f"Total {len(data)} samples")
    print(f"Sample: {json.dumps(data[0].model_dump(), ensure_ascii=False)}")

    with Session(engine) as session:
        session.add_all(data)
        session.commit()
        print("Upload complete.")


build_dataset(split="validation")
