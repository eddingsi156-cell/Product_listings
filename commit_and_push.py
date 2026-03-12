#!/usr/bin/env python3
"""提交并推送代码"""

import subprocess
import sys


def run_command(cmd):
    """运行命令并返回输出"""
    print(f"$ {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=r"c:\Users\16222\Documents\Product_listings"
        )
        print(result.stdout)
        if result.stderr:
            print("stderr:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    print("=== 提交并推送代码 ===")
    
    # 1. 查看状态
    print("\n--- git status ---")
    run_command("git status")
    
    # 2. 添加所有修改的文件
    print("\n--- git add -u ---")
    run_command("git add -u")
    
    # 3. 提交
    commit_msg = "修复滑块验证码：移除过冲逻辑 + 优化距离计算（减去 0.18*piece_width）"
    print(f"\n--- git commit -m '{commit_msg}' ---")
    success = run_command(f'git commit -m "{commit_msg}"')
    
    if not success:
        print("\n提交失败，可能没有变更需要提交")
        return 1
    
    # 4. 推送
    print("\n--- git push ---")
    success = run_command("git push")
    
    if success:
        print("\n✓ 代码已成功推送到远程仓库！")
        return 0
    else:
        print("\n✗ 推送失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
