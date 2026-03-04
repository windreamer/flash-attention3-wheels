#!/usr/bin/env python3
import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import jinja2
import requests

TEMPLATES_DIR = Path(__file__).parent / "templates"

WHEEL_RE = re.compile(
    r"flash_attn_3-(?P<base>\d+\.\d+\.\d+(?:[a-zA-Z]+\d*)?)"  # 3.0.0b1
    r"[.+](?P<date>\d{8})"  # 20250907
    r"[.+]cu(?P<cuda>\d{3})"  # 129
    r"torch(?P<torch>\d{3,4})"  # 280/2100
    r"cxx11abi(?P<abi>true|false)"  # true
    r"[.+][a-f0-9]+"  # dfb664
    r"-cp(?P<py>\d{2})-.+-(?P<platform>[a-z0-9_]+)\.whl",  # cp39-abi3-linux_x86_64.whl
    re.I,
)

CACHE_FILE = "download_stats.json"


class WheelIndexGenerator:
    def __init__(self, repo_owner: str, repo_name: str, token: str = None):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.token = token
        self.base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        self.headers = {}
        self.download_stats = {
            "file_stats": [],
            "total_downloads": 0,
            "total_daily_new": 0,
            "daily_new_stats": [],
        }
        self.yesterday_stats = {"file_stats": [], "total_downloads": 0}
        if token:
            self.headers["Authorization"] = f"token {token}"
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=True,
        )

    def get_releases(self) -> List[Dict]:
        """获取所有release"""
        url = f"{self.base_url}/releases"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def load_cached_stats(self, output_dir: str):
        """Load yesterday's stats from cache file"""
        cache_path = Path(output_dir) / CACHE_FILE
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    self.yesterday_stats = json.load(f)
                print(f"Loaded cached stats from {cache_path}")
            except Exception as e:
                print(f"Failed to load cache: {e}")
        else:
            print("No cached stats found, starting fresh")

    def calculate_download_stats(self, releases: List[Dict]):
        """Calculate download statistics from releases"""
        yesterday_files = {
            f"{item['file_name']}@{item['release']}": item
            for item in self.yesterday_stats.get("file_stats", [])
        }

        file_stats = []
        daily_new_stats = []
        total_downloads = 0
        yesterday_total = self.yesterday_stats.get("total_downloads", 0)

        for release in releases:
            release_name = release["tag_name"]
            for asset in release.get("assets", []):
                if not asset["name"].endswith(".whl"):
                    continue

                current_count = asset["download_count"]
                total_downloads += current_count

                file_key = f"{asset['name']}@{release_name}"
                yesterday_count = yesterday_files.get(file_key, {}).get(
                    "download_count", 0
                )
                daily_new = max(0, current_count - yesterday_count)

                file_info = {
                    "file_name": asset["name"],
                    "release": release_name,
                    "download_count": current_count,
                    "daily_new": daily_new,
                }
                file_stats.append(file_info)

                if daily_new > 0:
                    daily_new_stats.append(file_info)

        file_stats.sort(key=lambda x: x["download_count"], reverse=True)
        daily_new_stats.sort(key=lambda x: x["daily_new"], reverse=True)

        self.download_stats = {
            "file_stats": file_stats,
            "daily_new_stats": daily_new_stats,
            "total_downloads": total_downloads,
            "total_daily_new": total_downloads - yesterday_total,
        }

    def save_stats(self, output_dir: str):
        """Save stats for tomorrow's comparison"""
        cache_path = Path(output_dir) / CACHE_FILE
        with open(cache_path, "w") as f:
            json.dump(self.download_stats, f, indent=2)
        print(f"Saved stats to {cache_path}")

    def parse_wheel_info(self, filename: str) -> dict | None:
        m = WHEEL_RE.match(filename)
        if not m:
            return None
        return {
            "filename": filename,
            "base_version": m["base"],
            "build_date": m["date"],
            "cuda_version": m["cuda"],  # 12.3
            "torch_version": m["torch"],  # 2.4.0
            "cxx11_abi": m["abi"],  # TRUE / FALSE
            "python_version": f"{m['py'][0]}.{m['py'][1:]}",  # 310 -> 3.10
            "platform": m["platform"],  # linux_x86_64, win_amd64, linux_aarch64
        }

    def organize_wheels(self, releases: List[Dict]) -> Dict:
        """按CUDA和PyTorch版本组织wheels"""
        organized = {}

        for release in releases:
            release_date = release["published_at"][:10]
            for asset in release["assets"]:
                if asset["name"].endswith(".whl"):
                    info = self.parse_wheel_info(asset["name"])
                    if info:
                        cuda_ver = info["cuda_version"]
                        torch_ver = info["torch_version"]

                        key = (
                            (int(cuda_ver[:2]), int(cuda_ver[2:])),
                            (
                                int(torch_ver[:1]),
                                int(torch_ver[1:-1]),
                                int(torch_ver[-1:]),
                            ),
                        )
                        if key not in organized:
                            organized[key] = {
                                "wheels": [],
                                "tags": set(),
                                "index_name": f"cu{cuda_ver}_torch{torch_ver}",
                            }

                        organized[key]["wheels"].append(
                            {
                                "filename": info["filename"],
                                "download_url": asset["browser_download_url"],
                                "size": asset["size"],
                                "created_at": asset["created_at"],
                                "python_version": info["python_version"],
                                "flash_version": info["base_version"],
                                "release_date": release_date,
                                "cuda_version": cuda_ver,
                                "torch_version": torch_ver,
                            }
                        )

                        # 检测平台标签
                        platform = info["platform"]
                        if "win" in platform:
                            organized[key]["tags"].add("windows")
                        elif "aarch64" in platform or "arm64" in platform:
                            organized[key]["tags"].add("arm64")

        return organized

    def generate_simple_index(
        self, wheels: List[Dict], cuda_version: Tuple[int], torch_version: Tuple[int]
    ) -> str:
        """生成简单的HTML index页面"""
        template = self.jinja_env.get_template("simple_index.html.j2")
        return template.render(
            cuda_version=".".join(map(str, cuda_version)),
            torch_version=".".join(map(str, torch_version)),
            wheels=wheels,
            now=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

    def generate_main_index(self, organized_wheels: Dict) -> str:
        """生成主index页面"""
        template = self.jinja_env.get_template("index.html.j2")
        wheel_groups = []
        for key, group in sorted(organized_wheels.items(), reverse=True):
            wheels = group["wheels"]
            if not wheels:
                continue
            tags = group["tags"]
            cudaver = wheels[0]["cuda_version"]
            torch_ver = wheels[0]["torch_version"]
            wheel_groups.append(
                {
                    "index_name": group["index_name"],
                    "cuda_ver": f"{cudaver[:2]}.{cudaver[2:]}",
                    "torch_ver": f"{torch_ver[:1]}.{torch_ver[1:-1]}.{torch_ver[-1:]}",
                    "wheel_count": len(wheels),
                    "last_updated": max(w["release_date"] for w in wheels),
                    "has_windows": "windows" in tags,
                    "has_arm64": "arm64" in tags,
                }
            )
        return template.render(
            repo_owner=self.repo_owner,
            repo_name=self.repo_name,
            now=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            download_stats=self.download_stats,
            wheel_groups=wheel_groups,
        )

    def generate_all_pages(self, output_dir: str):
        """生成所有页面"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.load_cached_stats(output_dir)

        print("Fetching releases from GitHub...")
        releases = self.get_releases()
        self.calculate_download_stats(releases)
        print(f"Found {len(releases)} releases")

        print("Organizing wheels...")
        organized_wheels = self.organize_wheels(releases)

        self.save_stats(output_dir)
        # 生成主索引页面
        print("Generating main index page...")
        main_index = self.generate_main_index(organized_wheels)
        (output_path / "index.html").write_text(main_index)

        # 为每个组合生成索引页面
        for key, group in organized_wheels.items():
            wheels = group["wheels"]
            if not wheels:
                continue

            index_name = group["index_name"]

            print(f"Generating index page for {index_name}...")

            cuda_version, torch_version = key
            # 创建子目录
            subdir = output_path / index_name
            subdir.mkdir(exist_ok=True)

            # 生成索引页面
            index_content = self.generate_simple_index(
                wheels, cuda_version, torch_version
            )
            (subdir / "index.html").write_text(index_content)

        print(f"All pages generated in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyPI-like index pages for Flash-Attention wheels"
    )
    parser.add_argument("--owner", required=True, help="GitHub repository owner")
    parser.add_argument("--repo", required=True, help="GitHub repository name")
    parser.add_argument("--token", help="GitHub personal access token (optional)")
    parser.add_argument("--output", default="docs", help="Output directory")

    args = parser.parse_args()

    generator = WheelIndexGenerator(args.owner, args.repo, args.token)
    generator.generate_all_pages(args.output)


if __name__ == "__main__":
    main()
