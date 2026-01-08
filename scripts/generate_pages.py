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
    r"-cp(?P<py>\d{2})-.+-(?P<platform>[a-z0-9_]+)\.whl",  # cp39-abi3-linux_x86_64.whl
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

                        key = f"cu{cuda_ver}_torch{torch_ver}"
                        if key not in organized:
                            organized[key] = {"wheels": [], "tags": set()}

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
        self, wheels: List[Dict], cuda_version: str, torch_version: str
    ) -> str:
        """生成简单的HTML index页面"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Flash-Attention 3 Wheels - CUDA {cuda_version}, PyTorch {torch_version}</title>
    <meta name="api-version" value="2" />
    <script type="text/javascript">
        (function(c,l,a,r,i,t,y){{
            c[a]=c[a]||function(){{(c[a].q=c[a].q||[]).push(arguments)}};
            t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
            y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
        }})(window, document, "clarity", "script", "uy0pu9bh60");
    </script>
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
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 24px;
            background: #ffffff;
            color: #1a1a1a;
            line-height: 1.5;
        }
        .header {
            background: #f8f9fa;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 32px;
        }
        .header h1 {
            margin: 0 0 8px 0;
            font-size: 32px;
            font-weight: 700;
        }
        .header p {
            margin: 0;
            color: #666;
            font-size: 16px;
        }
        .update-banner {
            background: #e8f4fd;
            padding: 16px 20px;
            border-radius: 8px;
            margin: 24px 0 32px 0;
            border-left: 4px solid #007acc;
        }
        .update-banner h3 {
            margin: 0 0 4px 0;
            font-size: 18px;
            color: #007acc;
        }
        .update-banner p {
            margin: 0;
            font-size: 14px;
            color: #333;
        }
        h2 {
            font-size: 24px;
            margin: 24px 0 16px 0;
            font-weight: 600;
        }
        .wheel-grid { 
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
            max-width: 1200px;
        }
        .wheel-card {
            padding: 16px;
            background: #ffffff;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
            transition: all 0.2s;
            display: flex;
            flex-direction: column;
        }
        .wheel-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.12), 0 2px 4px rgba(0,0,0,0.08);
            transform: translateY(-2px);
        }
        .wheel-card h3 {
            margin: 0 0 8px 0;
            font-size: 16px;
            font-weight: 600;
            color: #1a1a1a;
            line-height: 1.3;
        }
        .wheel-card .tags {
            margin-bottom: 8px;
            min-height: 22px;
        }
        .wheel-card .stats {
            color: #666;
            font-size: 12px;
            margin: 0 0 12px 0;
            flex-grow: 1;
        }
        .wheel-link {
            display: block;
            text-align: center;
            padding: 8px 12px;
            background: #007acc;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            transition: background-color 0.2s;
        }
        .wheel-link:hover {
            background: #005a9a;
        }
        .windows-badge, .arm64-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
            margin-right: 6px;
        }
        .windows-badge {
            background: linear-gradient(135deg, #007acc, #005a9a);
            color: white;
        }
        .arm64-badge {
            background: linear-gradient(135deg, #00c853, #009624);
            color: white;
        }
        details { margin-top: 10px; }
        summary {
            font-size: 12px;
            color: #666;
            cursor: pointer;
        }
        summary:hover { color: #007acc; }
        details code {
            display: block;
            margin-top: 6px;
            padding: 10px;
            font-size: 11px;
            border-radius: 6px;
            background: #f8f9fa;
            border: 1px solid #e0e0e0;
            white-space: pre-wrap;
            word-break: break-all;
        }
        pre { margin: 10px 0; }
        code {
            background: #f8f9fa;
            padding: 2px 5px;
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
            font-size: 12px;
        }
        ul { padding-left: 20px; }
        p { margin: 8px 0; }
        .copy-btn { position: absolute; top: 6px; right: 6px; background: none; border: none; cursor: pointer; padding: 4px; font-size: 16px; z-index: 1; }
        .copy-btn:hover { opacity: 0.7; }
    </style>
    <script type="text/javascript">
        (function(c,l,a,r,i,t,y){
            c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
            t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
            y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
        })(window, document, "clarity", "script", "uy0pu9bh60");
    </script>
</head>
<body>
    <div class="header">
        <h1>🔥 Flash-Attention 3 Wheels Repository</h1>
        <p>Pre-built wheels for Flash-Attention 3, updated weekly</p>
        <p>Generated on: """
            + datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            + """</p>
    </div>

    <div class="update-banner">
        <h3>🚀 Windows Wheels Now Available!</h3>
        <p>We've successfully built Flash Attention 3 wheels for <strong>Windows</strong> (CUDA 12 only for now).</p>
    </div>

    <h2>Installation Instructions</h2>
    <p>Add the appropriate index URL to your pip command:</p>
    <pre><code>pip install flash_attn_3 --find-links https://"""
            + f"{self.repo_owner}.github.io/{self.repo_name}"
            + """/PATH/TO/INDEX</code></pre>

    <h2>Available Wheel Indexes</h2>
    <div class="wheel-grid">
"""
        )

        for key, group in sorted(organized_wheels.items(), reverse=True):
            wheels = group["wheels"]
            if not wheels:
                continue

            tags = group["tags"]
            cudaver = wheels[0]["cuda_version"]
            cuda_ver = f"{cudaver[:2]}.{cudaver[2:]}"
            torch_ver = wheels[0]["torch_version"]
            torch_ver = f"{torch_ver[0]}.{torch_ver[1]}.{torch_ver[2:]}"

            wheel_count = len(wheels)
            last_updated = max(w["release_date"] for w in wheels)

            # 生成标签HTML
            tags_html = ""
            if "windows" in tags:
                tags_html += "<span class='windows-badge'>Windows Support</span>"
            if "arm64" in tags:
                tags_html += "<span class='arm64-badge'>Arm64 Support</span>"

            html += f"""
        <div class="wheel-card">
            <h3>CUDA {cuda_ver}, PyTorch {torch_ver}</h3>
            <div class="tags">{tags_html}</div>
            <p class="stats">{wheel_count} wheels available • Last updated: {last_updated}</p>
            <a href="{key}/index.html" class="wheel-link">View Wheels</a>
            <div class="pip-command" style="position: relative; margin-top: 10px;">
                <details>
                    <summary style="font-size: 12px; color: #666; cursor: pointer;">Direct pip command</summary>
                    <div style="position: relative;">
                        <button onclick="copyPipCommand(this)" class="copy-btn" title="Copy command">📋</button>
                        <code style="display: block; margin-top: 6px; padding: 10px; padding-right: 30px; font-size: 11px; border-radius: 6px; background: #f8f9fa; border: 1px solid #e0e0e0; white-space: pre-wrap; word-break: break-all;">pip install flash_attn_3 --find-links https://{self.repo_owner}.github.io/{self.repo_name}/{key} --extra-index-url https://download.pytorch.org/whl/cu{cudaver}</code>
                    </div>
                </details>
            </div>
        </div>
"""

        html += (
            """
    </div>

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

    <script>
        function copyPipCommand(button) {
            const code = button.parentElement.querySelector('code');
            navigator.clipboard.writeText(code.textContent).then(() => {
                button.textContent = '✓';
                setTimeout(() => {
                    button.textContent = '📋';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy: ', err);
                alert('Copy failed, please copy manually');
            });
        }
    </script>
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
        for key, group in organized_wheels.items():
            wheels = group["wheels"]
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
