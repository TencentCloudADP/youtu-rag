"""
Clear error cache files
"""

import json
import os
import pathlib
import subprocess

from tqdm import tqdm

cache_dir = pathlib.Path(__file__).parent.parent.joinpath(".cache").resolve()

need_delete_names = []
for file_name in tqdm(os.listdir(cache_dir)):
    if not file_name.endswith(".json"):
        continue
    data = json.load(open(f"{cache_dir}/{file_name}"))
    if "result" not in data:
        need_delete_names.append(file_name)
    else:
        result = data["result"]
        if "InsufficientBalanceError" in result or "TimeoutError" in result or len(result) < 100:
            need_delete_names.append(file_name)

print("Found errors file cnt: ", len(need_delete_names))
print(need_delete_names[:20])

print("File cnt before delete", len(os.listdir(cache_dir)))
os.makedirs(f"{cache_dir}/rubbish_bin/", exist_ok=True)

if_clear = input("Clear cache? (y/n): ")
if if_clear == "y":
    for file_name in need_delete_names:
        subprocess.run(["mv", f"{cache_dir}/{file_name}", f"{cache_dir}/rubbish_bin/"])
    print("File cnt after delete", len(os.listdir(cache_dir)))
