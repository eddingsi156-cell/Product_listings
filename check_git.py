#!/usr/bin/env python3
"""检查 git 状态"""

import subprocess
import sys


def run_cmd(cmd):
    """运行命令"""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=r"c:\Users\16222\Documents\Product_listings"
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def main():
    print("=== 检查 Git 状态 ===")
    
    # 1. 检查当前分支
    print("\n1. 当前分支:")
    code, out, err = run_cmd("git branch --show-current")
    if code == 0:
        print(f"   {out}")
    
    # 2. 检查状态
    print("\n2. Git 状态:")
    code, out, err = run_cmd("git status --porcelain")
    if code == 0:
        if out:
            print("   有未提交的变更:")
            for line in out.split('\n'):
                print(f"   {line}")
        else:
            print("   工作区干净")
    
    # 3. 检查最后一次提交
    print("\n3. 最后一次提交:")
    code, out, err = run_cmd("git log -1 --oneline")
    if code == 0:
        print(f"   {out}")
    
    # 4. 检查与远程的差异
    print("\n4. 本地与远程的差异:")
    code, out, err = run_cmd("git log origin/master..HEAD --oneline")
    if code == 0:
        if out:
            print("   有未推送的提交:")
            for line in out.split('\n'):
                print(f"   {line}")
        else:
            print("   本地与远程同步")
    
    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
