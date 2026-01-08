import json
import os
import re

import requests

url = "https://hub.docker.com/v2/repositories/nvidia/cuda/tags/"
params = {"page_size": 100, "name": "ubuntu22.04"}
tags = []
while url:
    resp = requests.get(url, params=params).json()
    tags.extend(resp["results"])
    url = resp.get("next")
pattern = re.compile(r"^(\d+)\.(\d+)\.(\d+)-devel-ubuntu22\.04$")
matches = []
for t in tags:
    m = pattern.match(t["name"])
    if m:
        major, minor, patch = m.groups()
        matches.append((f"{major}.{minor}", int(patch)))
latest = {}
for major_minor, patch in matches:
    key = major_minor
    if key not in latest or patch > latest[key][1]:
        latest[key] = (major_minor, patch)

cuda_full_map = {k: f"{k}.{v[1]}" for k, v in latest.items()}
target = os.getenv("MATRIX_TARGET", "linux").lower()

if target == 'windows':
    cuda_versions = ["12.8", "12.9"]
    torch_versions = ["2.9.0"]
elif target == 'arm':
    cuda_versions = ["12.9", "13.0"]
    torch_versions = ["2.8.0", "2.9.1"]
else:
    cuda_versions = ["12.8", "12.9",  "13.0"]
    torch_versions = ["2.8.0", "2.9.1"]


BLACKLIST = {
    "2.8": "12.9",
}


def ver2tuple(v: str):
    return tuple(map(int, v.split(".")))

matrix = {"include": []}

for cuda in cuda_versions:
    cuda = cuda.strip()
    assert cuda in cuda_full_map
    cuda_full = cuda_full_map[cuda]
    for torch in torch_versions:
        torch_major = torch.rsplit(".", 1)[0]
        if torch_major in BLACKLIST:
            max_cuda = BLACKLIST[torch_major]
            if ver2tuple(cuda) > ver2tuple(max_cuda):
                continue

        matrix["include"].append(
            {
                "cuda": cuda,
                "cuda_full": cuda_full,
                "torch": torch.strip(),
            }
        )

matrix_json = json.dumps(matrix, separators=(",", ":"))

if "GITHUB_OUTPUT" in os.environ:
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"matrix<<EOF\n{matrix_json}\nEOF\n")

print("Generated matrix:")
print(json.dumps(matrix, indent=2))
