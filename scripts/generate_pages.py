#!/usr/bin/env python3
import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import requests

WHEEL_RE = re.compile(
    r"flash_attn_3-(?P<base>\d+\.\d+\.\d+(?:[a-zA-Z]+\d*)?)"  # 3.0.0b1
    r"[.+](?P<date>\d{8})"  # 20250907
    r"[.+]cu(?P<cuda>\d{3})"  # 129
    r"torch(?P<torch>\d{3})"  # 280
    r"cxx11abi(?P<abi>true|false)"  # true
    r"[.+][a-f0-9]+"  # dfb664
    r"-cp(?P<py>\d{2})-.*\.whl",  # cp39-abi3-linux_x86_64.whl
    re.I,
)


class WheelIndexGenerator:
    def __init__(self, repo_owner: str, repo_name: str, token: str = None):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.token = token
        self.base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        self.headers = {}
        if token:
            self.headers["Authorization"] = f"token {token}"

    def get_releases(self) -> List[Dict]:
        """获取所有release"""
        url = f"{self.base_url}/releases"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

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

                        key = f"cu{cuda_ver}_torch{torch_ver}"
                        if key not in organized:
                            organized[key] = []

                        organized[key].append(
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

        return organized

    def generate_simple_index(
        self, wheels: List[Dict], cuda_version: str, torch_version: str
    ) -> str:
        """生成简单的HTML index页面"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Flash-Attention 3 Wheels - CUDA {cuda_version}, PyTorch {torch_version}</title>
    <meta name="api-version" value="2" />
</head>
<body>
    <h1>Flash-Attention 3 Wheels</h1>
    <p>CUDA {cuda_version}, PyTorch {torch_version}</p>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>

    <ul>
"""

        for wheel in sorted(wheels, key=lambda x: x["filename"]):
            html += f'        <li><a href="{wheel["download_url"]}">{wheel["filename"]}</a></li>\n'

        html += """    </ul>
</body>
</html>"""
        return html

    def generate_main_index(self, organized_wheels: Dict) -> str:
        """生成主index页面"""
        html = (
            """<!DOCTYPE html>
<html>
<head>
    <title>Flash-Attention 3 Wheels Repository</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f4f4f4; padding: 20px; border-radius: 5px; }
        .wheel-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .wheel-link { display: inline-block; margin: 5px; padding: 8px 12px; background: #007acc; color: white; text-decoration: none; border-radius: 3px; }
        .wheel-link:hover { background: #005a9a; }
        .stats { color: #666; font-size: 0.9em; }
        .windows-badge { background: linear-gradient(135deg, #007acc, #005a9a); color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.75em; font-weight: 600; }
        code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔥 Flash-Attention 3 Wheels Repository</h1>
        <p>Pre-built wheels for Flash-Attention 3, updated weekly</p>
        <p>Generated on: """
            + datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            + """</p>
        </div>
    </div>

    <div style="background: #e8f4fd; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h2>Update</h2>
        <h3>🚀 Windows Wheels Now Available!</h3>
        <p>We've successfully built Flash Attention 3 wheels for <strong>Windows</strong> (CUDA 12 only for now).</p>
    </div>

    <h2>Installation Instructions</h2>
    <p>Add the appropriate index URL to your pip command:</p>
    <pre><code>pip install flash_attn_3 --find-links https://"""
            + f"{self.repo_owner}.github.io/{self.repo_name}"
            + """/PATH/TO/INDEX</code></pre>

    <h2>Available Wheel Indexes</h2>
"""
        )

        for key, wheels in sorted(organized_wheels.items(), reverse=True):
            if not wheels:
                continue

            cudaver = cuda_ver = wheels[0]["cuda_version"]
            cuda_ver = f"{cuda_ver[:2]}.{cuda_ver[2:]}"
            torch_ver = wheels[0]["torch_version"]
            torch_ver = f"{torch_ver[0]}.{torch_ver[1]}.{torch_ver[2:]}"

            wheel_count = len(wheels)
            last_updated = max(w["release_date"] for w in wheels)

            html += f"""
    <div class="wheel-section">
        <h3>CUDA {cuda_ver}, PyTorch {torch_ver}</h3>
        {"<span class='windows-badge'>Windows Support</span>" if cuda_ver in ["12.8", "12.9"] else ""}
        <p class="stats">{wheel_count} wheels available • Last updated: {last_updated}</p>
        <a href="{key}/index.html" class="wheel-link">View Wheels</a>
        <details>
            <summary>Direct pip command</summary>
            <code>pip install flash_attn_3 --find-links https://{self.repo_owner}.github.io/{self.repo_name}/{key} --extra-index-url https://download.pytorch.org/whl/cu{cudaver} </code>
        </details>
    </div>
"""

        html += (
            """
    <h2>Quick Reference</h2>
    <ul>
        <li><strong>GitHub Repository:</strong> <a href="https://github.com/"""
            + f"{self.repo_owner}/{self.repo_name}"
            + """">https://github.com/"""
            + f"{self.repo_owner}/{self.repo_name}"
            + """</a></li>
        <li><strong>Build Schedule:</strong> Weekly (Sundays at 2 AM UTC)</li>
    </ul>

    <h2>Usage Examples</h2>
    <pre><code># Install for CUDA 12.3, PyTorch 2.4.0
pip install flash_attn_3 --find-links https://"""
            + f"{self.repo_owner}.github.io/{self.repo_name}"
            + """/cu128_torch280

# Install specific version
pip install flash_attn_3==3.0.0 --find-links https://"""
            + f"{self.repo_owner}.github.io/{self.repo_name}"
            + """/cu128_torch280

# Upgrade existing installation
pip install --upgrade flash_attn_3 --find-links https://"""
            + f"{self.repo_owner}.github.io/{self.repo_name}"
            + """/cu128_torch280</code></pre>
</body>
</html>"""
        )

        return html

    def generate_all_pages(self, output_dir: str):
        """生成所有页面"""
        print("Fetching releases from GitHub...")
        releases = self.get_releases()
        print(f"Found {len(releases)} releases")

        print("Organizing wheels...")
        organized_wheels = self.organize_wheels(releases)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 生成主索引页面
        print("Generating main index page...")
        main_index = self.generate_main_index(organized_wheels)
        (output_path / "index.html").write_text(main_index)

        # 为每个组合生成索引页面
        for key, wheels in organized_wheels.items():
            if not wheels:
                continue

            print(f"Generating index page for {key}...")

            cuda_version, torch_version = key.split("_")
            # 创建子目录
            subdir = output_path / key
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
