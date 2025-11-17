#!/usr/bin/env python3
"""
初始化脚本：下载Kaggle数据并设置环境
1. 从Kaggle下载竞赛数据并解压
2. 创建data/目录并移动必要文件
3. 删除多余文件
4. 安装environment.txt中的依赖包
"""

import os
import sys
import subprocess
import shutil
import zipfile
import platform
from pathlib import Path


def get_pip_command():
    """根据操作系统返回合适的pip命令"""
    system = platform.system()
    python_cmd = sys.executable
    
    # 尝试使用当前Python解释器的pip
    if python_cmd:
        return f"{python_cmd} -m pip"
    
    # 回退方案
    if system == "Windows":
        return "pip"
    else:
        return "pip3"


def install_packages():
    """安装environment.txt中的包"""
    print("=" * 60)
    print("Installing packages from environment.txt...")
    print("=" * 60)
    
    pip_cmd = get_pip_command()
    env_file = Path(__file__).parent / "environment.txt"
    
    if not env_file.exists():
        print(f"Error: {env_file} not found!")
        return False
    
    try:
        # 读取包列表
        with open(env_file, 'r') as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"Found {len(packages)} packages to install:")
        for pkg in packages:
            print(f"  - {pkg}")
        
        # 安装包
        cmd = f"{pip_cmd} install {' '.join(packages)}"
        print(f"\nRunning: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        
        print("\nInstallation completed successfully!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nError installing packages:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False


def download_kaggle_data():
    """下载Kaggle竞赛数据"""
    print("=" * 60)
    print("Downloading Kaggle competition data...")
    print("=" * 60)
    
    competition_name = "influencers-or-observers-predicting-social-roles"
    
    try:
        # 检查kaggle是否已安装
        result = subprocess.run(["kaggle", "--version"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("Error: kaggle CLI not found. Please install it first:")
            print("  pip install kaggle")
            print("  Then configure with: kaggle datasets download -h")
            return False
        
        # 下载数据
        project_root = Path(__file__).parent
        os.chdir(project_root)
        
        cmd = f"kaggle competitions download -c {competition_name}"
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True,
                              capture_output=True, text=True)
        
        print("Download completed!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading data:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("Error: kaggle CLI not found. Please install it first:")
        print("  pip install kaggle")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def extract_and_organize_files():
    """解压文件并整理到data/目录"""
    print("=" * 60)
    print("Extracting and organizing files...")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    
    # 创建data目录
    data_dir.mkdir(exist_ok=True)
    print(f"Created/verified data directory: {data_dir}")
    
    # 查找zip文件
    zip_files = list(project_root.glob("*.zip"))
    
    if not zip_files:
        print("No zip files found. Skipping extraction.")
        return True
    
    # 解压所有zip文件
    for zip_file in zip_files:
        print(f"\nExtracting {zip_file.name}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(project_root)
            print(f"  Extracted {zip_file.name}")
        except Exception as e:
            print(f"  Error extracting {zip_file.name}: {e}")
            continue
    
    # 查找并移动目标文件
    target_files = ["kaggle_test.jsonl", "train.jsonl"]
    moved_files = []
    
    for target_file in target_files:
        # 在项目根目录查找
        source_path = project_root / target_file
        dest_path = data_dir / target_file
        
        if source_path.exists():
            if dest_path.exists():
                print(f"  {target_file} already exists in data/, skipping...")
            else:
                shutil.move(str(source_path), str(dest_path))
                print(f"  Moved {target_file} to data/")
                moved_files.append(target_file)
        else:
            print(f"  Warning: {target_file} not found after extraction")
    
    # 清理多余文件
    print("\nCleaning up extra files...")
    
    # 删除zip文件
    for zip_file in zip_files:
        try:
            zip_file.unlink()
            print(f"  Deleted {zip_file.name}")
        except Exception as e:
            print(f"  Error deleting {zip_file.name}: {e}")
    
    # 删除其他可能的多余文件（保留项目结构）
    files_to_remove = []
    for item in project_root.iterdir():
        if item.is_file():
            # 保留重要的项目文件
            if item.name in ["initiate.py", "environment.txt", ".gitignore", "README.md"]:
                continue
            # 删除其他文件（可能是解压出来的临时文件）
            if item.suffix in [".zip", ".csv", ".json", ".jsonl"] and item.parent == project_root:
                # 但不要删除已经在data/目录中的文件
                if item.name not in target_files:
                    files_to_remove.append(item)
    
    for file_to_remove in files_to_remove:
        try:
            # 再次检查是否在data/目录中
            if not (data_dir / file_to_remove.name).exists():
                file_to_remove.unlink()
                print(f"  Deleted {file_to_remove.name}")
        except Exception as e:
            print(f"  Error deleting {file_to_remove.name}: {e}")
    
    print("\nFile organization completed!")
    return True


def main():
    """主函数"""
    print("=" * 60)
    print("Kaggle Competition Setup Script")
    print("=" * 60)
    print()
    
    # 1. 安装依赖包
    print("Step 1: Installing dependencies...")
    if not install_packages():
        print("Warning: Package installation had issues. Continuing anyway...")
    print()
    
    # 2. 下载Kaggle数据
    print("Step 2: Downloading Kaggle data...")
    if not download_kaggle_data():
        print("Error: Failed to download data. Please check your Kaggle API credentials.")
        print("You can set them up at: https://www.kaggle.com/account")
        return
    print()
    
    # 3. 解压和组织文件
    print("Step 3: Extracting and organizing files...")
    if not extract_and_organize_files():
        print("Error: Failed to organize files.")
        return
    print()
    
    # 验证最终结果
    print("=" * 60)
    print("Verification:")
    print("=" * 60)
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    
    required_files = ["kaggle_test.jsonl", "train.jsonl"]
    all_present = True
    
    for file_name in required_files:
        file_path = data_dir / file_name
        if file_path.exists():
            print(f"✓ {file_name} found in data/")
        else:
            print(f"✗ {file_name} NOT found in data/")
            all_present = False
    
    print()
    if all_present:
        print("=" * 60)
        print("Setup completed successfully!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("Setup completed with warnings. Please check the files manually.")
        print("=" * 60)


if __name__ == "__main__":
    main()

