import argparse
import json

import pandas as pd

from utu.config import ConfigLoader, EvalConfig
from utu.eval.data import DBDataManager


def get_stat(config: EvalConfig):
    db = DBDataManager(config)
    samples = db.get_samples()
    df = pd.DataFrame([sample.as_dict() for sample in samples])
    # stat data group by stage
    stat = df.groupby("stage").size()
    print(stat)


def dump_exp_samples(config: EvalConfig, output_file: str, clear_records: bool = False, limit: int = None):
    db = DBDataManager(config)
    samples = db.get_samples(limit=limit)
    print(f"Found {len(samples)} samples in db.")
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.as_dict(), ensure_ascii=False) + "\n")
    if clear_records:
        print(f"Clearing {len(samples)} samples from db.")
        db.delete_samples(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="ww", help="Configuration name for evaluation.")
    parser.add_argument("--op", type=str, choices=["stat", "dump"], default="dump", help="Option to run.")
    parser.add_argument("--exp_id", type=str, default=None, help="Experiment ID.")
    parser.add_argument("--output_file", type=str, default=None, help="Output file.")
    parser.add_argument("--clear_records", action="store_true", help="Clear records from db.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to dump.")
    args = parser.parse_args()

    config = ConfigLoader.load_eval_config(args.config_name)
    if args.exp_id:
        config.exp_id = args.exp_id

    if not args.output_file:
        args.output_file = f"data/output_{config.exp_id}.jsonl"

    match args.op:
        case "stat":
            get_stat(config)
        case "dump":
            dump_exp_samples(config, output_file=args.output_file, clear_records=args.clear_records, limit=args.limit)


if __name__ == "__main__":
    main()
