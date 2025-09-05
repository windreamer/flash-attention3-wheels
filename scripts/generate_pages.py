#!/usr/bin/env python3
import json
import os
import re
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import argparse

# match flash_attn-3.0.0+20240905+cu12torch2.4cxx11abiFALSE+g123456-cp310-cp310-linux_x86_64.whl
WHEEL_RE = re.compile(
    r"flash_attn-(?P<base>\d+\.\d+\.\d+)"          # 3.0.0
    r"\+(?P<date>\d{8})\+cu(?P<cuda>\d+\.\d+)"     # 20240905+cu12.3
    r"torch(?P<torch>\d+\.\d+\.\d+)"                # torch2.4.0
    r"cxx11abi(?P<abi>TRUE|FALSE)"                  # cxx11abiFALSE
    r"\+g[a-f0-9]+"                                  # +g123456
    r"-cp(?P<py>\d{2})-.*linux.*\.whl"              # -cp310-cp310-linux_x86_64.whl
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
            "cuda_version": m["cuda"],          # 12.3
            "torch_version": m["torch"],        # 2.4.0
            "cxx11_abi": m["abi"],              # TRUE / FALSE
            "python_version": f"{m['py'][0]}.{m['py'][1:]}",  # 310 -> 3.10
        }
    
    def organize_wheels(self, releases: List[Dict]) -> Dict:
        """按CUDA和PyTorch版本组织wheels"""
        organized = {}
        
        for release in releases:
            release_date = release['published_at'][:10]
            for asset in release['assets']:
                if asset['name'].endswith('.whl'):
                    info = self.parse_wheel_info(asset['name'])
                    if info:
                        cuda_ver = info['cuda_version']
                        torch_ver = info['torch_version']
                        
                        key = f"cu{cuda_ver.replace('.', '')}_torch{torch_ver.replace('.', '')}"
                        if key not in organized:
                            organized[key] = []
                        
                        organized[key].append({
                            'filename': info['filename'],
                            'download_url': asset['browser_download_url'],
                            'size': asset['size'],
                            'created_at': asset['created_at'],
                            'python_version': info['python_version'],
                            'flash_version': info['version'],
                            'release_date': release_date
                        })
        
        return organized
    
    def generate_simple_index(self, wheels: List[Dict], cuda_version: str, torch_version: str) -> str:
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
        
        for wheel in sorted(wheels, key=lambda x: x['filename']):
            html += f'        <li><a href="{wheel["download_url"]}">{wheel["filename"]}</a></li>\n'
        
        html += """    </ul>
</body>
</html>"""
        return html
    
    def generate_main_index(self, organized_wheels: Dict) -> str:
        """生成主index页面"""
        html = """<!DOCTYPE html>
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
        code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔥 Flash-Attention 3 Wheels Repository</h1>
        <p>Pre-built wheels for Flash-Attention 3, updated weekly</p>
        <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC') + """</p>
    </div>
    
    <h2>Installation Instructions</h2>
    <p>Add the appropriate index URL to your pip command:</p>
    <pre><code>pip install flash-attn3 --find-links https://""" + f"{self.repo_owner}.github.io/{self.repo_name}" + """/PATH/TO/INDEX</code></pre>
    
    <h2>Available Wheel Indexes</h2>
"""
        
        for key, wheels in sorted(organized_wheels.items()):
            if not wheels:
                continue
                
            cuda_ver = wheels[0]['download_url'].split('cu')[1].split('-')[0]
            cuda_ver = f"{cuda_ver[0]}.{cuda_ver[1:]}"
            torch_ver = wheels[0]['download_url'].split('torch')[1].split('-')[0]
            torch_ver = f"{torch_ver[0]}.{torch_ver[1:3]}.{torch_ver[3:]}"
            
            wheel_count = len(wheels)
            last_updated = max(w['release_date'] for w in wheels)
            
            html += f"""
    <div class="wheel-section">
        <h3>CUDA {cuda_ver}, PyTorch {torch_version}</h3>
        <p class="stats">{wheel_count} wheels available • Last updated: {last_updated}</p>
        <a href="{key}/index.html" class="wheel-link">View Wheels</a>
        <details>
            <summary>Direct pip command</summary>
            <code>pip install flash-attn3 --find-links https://{self.repo_owner}.github.io/{self.repo_name}/{key}</code>
        </details>
    </div>
"""
        
        html += """
    <h2>Quick Reference</h2>
    <ul>
        <li><strong>GitHub Repository:</strong> <a href="https://github.com/""" + f"{self.repo_owner}/{self.repo_name}" + """">https://github.com/""" + f"{self.repo_owner}/{self.repo_name}" + """</a></li>
        <li><strong>Build Schedule:</strong> Weekly (Sundays at 2 AM UTC)</li>
        <li><strong>Supported CUDA Versions:</strong> 12.3, 12.4, 12.5+</li>
        <li><strong>Supported PyTorch Versions:</strong> 2.3.0, 2.4.0, 2.5.0+</li>
        <li><strong>Supported Python Versions:</strong> 3.9, 3.10, 3.11, 3.12</li>
    </ul>
    
    <h2>Usage Examples</h2>
    <pre><code># Install for CUDA 12.3, PyTorch 2.4.0
pip install flash-attn3 --find-links https://""" + f"{self.repo_owner}.github.io/{self.repo_name}" + """/cu123_torch240

# Install specific version
pip install flash-attn3==3.0.0 --find-links https://""" + f"{self.repo_owner}.github.io/{self.repo_name}" + """/cu123_torch240

# Upgrade existing installation
pip install --upgrade flash-attn3 --find-links https://""" + f"{self.repo_owner}.github.io/{self.repo_name}" + """/cu123_torch240</code></pre>
</body>
</html>"""
        
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
            
            # 从第一个wheel获取版本信息
            sample_wheel = wheels[0]
            cuda_match = re.search(r'cu(\d+\.\d+)', sample_wheel['download_url'])
            torch_match = re.search(r'torch(\d+\.\d+\.\d+)', sample_wheel['download_url'])
            
            if cuda_match and torch_match:
                cuda_version = cuda_match.group(1)
                torch_version = torch_match.group(1)
                
                # 创建子目录
                subdir = output_path / key
                subdir.mkdir(exist_ok=True)
                
                # 生成索引页面
                index_content = self.generate_simple_index(wheels, cuda_version, torch_version)
                (subdir / "index.html").write_text(index_content)
        
        print(f"All pages generated in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate PyPI-like index pages for Flash-Attention wheels")
    parser.add_argument("--owner", required=True, help="GitHub repository owner")
    parser.add_argument("--repo", required=True, help="GitHub repository name")
    parser.add_argument("--token", help="GitHub personal access token (optional)")
    parser.add_argument("--output", default="docs", help="Output directory")
    
    args = parser.parse_args()
    
    generator = WheelIndexGenerator(args.owner, args.repo, args.token)
    generator.generate_all_pages(args.output)

if __name__ == "__main__":
    main()
