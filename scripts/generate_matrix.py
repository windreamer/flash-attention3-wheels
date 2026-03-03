import json
import os
import re
from collections import defaultdict

import requests

DOCKER_TAGS_API = "https://hub.docker.com/v2/repositories/nvidia/cuda/tags/"
PYTORCH_WHL_INDEX = "https://download.pytorch.org/whl/torch/"


def get_latest_cuda_patches_for_ubuntu2204() -> dict[str, str]:
    """
    Return mapping like {"12.8": "12.8.1", "12.9": "12.9.0"} by scanning
    nvidia/cuda tags that match: <major>.<minor>.<patch>-devel-ubuntu22.04
    """
    tag_re = re.compile(r"^(\d+)\.(\d+)\.(\d+)-devel-ubuntu22\.04$")
    url = DOCKER_TAGS_API
    params = {"page_size": 100, "name": "ubuntu22.04"}

    latest_patch: dict[str, int] = {}  # key: "major.minor" -> max patch int

    while url:
        resp = requests.get(url, params=params).json()
        for item in resp.get("results", []):
            name = item.get("name", "")
            m = tag_re.match(name)
            if not m:
                continue
            major, minor, patch = m.groups()
            key = f"{major}.{minor}"
            latest_patch[key] = max(latest_patch.get(key, -1), int(patch))
        url = resp.get("next")

    return {k: f"{k}.{p}" for k, p in latest_patch.items()}


def get_target_versions(target: str) -> tuple[list[str], list[str]]:
    """Return (cuda_versions, torch_versions) for a given target platform."""
    target = (target or "linux").lower()

    if target == "windows":
        cuda_versions = ["12.8", "12.9"]
        torch_versions = ["2.8.0", "2.9.1"]
    elif target == "arm":
        cuda_versions = ["12.8", "12.9", "13.0"]
        torch_versions = ["2.8.0", "2.10.0"]
    else:
        cuda_versions = ["12.8", "12.9", "13.0"]
        torch_versions = ["2.8.0", "2.10.0"]

    return cuda_versions, torch_versions


def fetch_html(url: str) -> str:
    r = requests.get(url, timeout=30, headers={"User-Agent": "python-requests"})
    r.raise_for_status()
    return r.text


def extract_wheel_names(html: str) -> list[str]:
    # The page is a simple directory index; links contain *.whl.
    return re.findall(r"torch-[^>%]*\.whl", html)


def platform_bucket(filename: str) -> str | None:
    # Example: torch-2.2.2+cu121-cp310-cp310-linux_x86_64.whl
    if filename.endswith("_x86_64.whl"):
        return "linux"
    if filename.endswith("_aarch64.whl") or filename.endswith("_arm64.whl"):
        return "arm"
    if filename.endswith("win_amd64.whl"):
        return "windows"
    return None


def parse_torch_version_and_cuda(filename: str) -> tuple[str, str] | None:
    """
    Extract torch version and CUDA tag (+cu118/+cu121/+cu124/...).
    Keep only cp310 wheels with CUDA builds; exclude +cpu.
    """
    if "-cp310-" not in filename or "+cu" not in filename or "+cpu" in filename:
        return None

    m = re.match(r"^torch-([0-9][^+]+)\+(cu\d+)-cp310-", filename)
    if not m:
        return None
    return m.group(1), m.group(2)


def build_pytorch_cuda_table() -> dict[str, dict[str, set[str]]]:
    """
    Build table: platform -> torch_version -> {cuda_versions}
    Only keep torch >= 2.8.0 and cuda >= 12.6.
    """
    html = fetch_html(PYTORCH_WHL_INDEX)
    wheels = extract_wheel_names(html)

    table: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for whl in wheels:
        plat = platform_bucket(whl)
        if not plat:
            continue

        parsed = parse_torch_version_and_cuda(whl)
        if not parsed:
            continue

        torch_ver, cuda_tag = parsed

        torch_ver_tuple = tuple(map(int, torch_ver.split(".")))
        cuda_ver_tuple = (int(cuda_tag[2:4]), int(cuda_tag[4:]))

        if torch_ver_tuple < (2, 8, 0) or cuda_ver_tuple < (12, 6):
            continue

        torch_ver_norm = ".".join(map(str, torch_ver_tuple))
        cuda_ver_norm = ".".join(map(str, cuda_ver_tuple))
        table[plat][torch_ver_norm].add(cuda_ver_norm)

    return table


def write_github_output(matrix_json: str) -> None:
    if "GITHUB_OUTPUT" not in os.environ:
        return
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"matrix<<EOF\n{matrix_json}\nEOF\n")


def main() -> None:
    cuda_full_map = get_latest_cuda_patches_for_ubuntu2204()

    target = os.getenv("MATRIX_TARGET", "linux").lower()
    cuda_versions, torch_versions = get_target_versions(target)

    pytorch_table = build_pytorch_cuda_table()
    target_filter = pytorch_table.get(target, {})

    matrix = {"include": []}

    for cuda in cuda_versions:
        cuda = cuda.strip()
        assert cuda in cuda_full_map
        cuda_full = cuda_full_map[cuda]

        for torch in torch_versions:
            torch = torch.strip()
            if cuda not in target_filter[torch]:
                continue
            matrix["include"].append(
                {"cuda": cuda, "cuda_full": cuda_full, "torch": torch}
            )

    matrix_json = json.dumps(matrix, separators=(",", ":"))
    write_github_output(matrix_json)

    print("Generated matrix:")
    print(json.dumps(matrix, indent=2))


if __name__ == "__main__":
    main()
