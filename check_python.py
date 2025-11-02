#!/usr/bin/env python3
"""
Check which Python and pip commands are available on this system.
Run this script first to determine the correct commands to use.
"""

import sys
import subprocess
import shutil

def check_command(cmd):
    """Check if a command exists and is executable."""
    return shutil.which(cmd) is not None

def get_python_version(cmd):
    """Get Python version for a command."""
    try:
        result = subprocess.run([cmd, '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None

def main():
    print("=" * 60)
    print("Python Environment Check")
    print("=" * 60)
    
    # Check Python commands
    python_cmds = ['python3', 'python']
    python_cmd = None
    
    print("\n[1] Checking Python commands...")
    for cmd in python_cmds:
        if check_command(cmd):
            version = get_python_version(cmd)
            if version:
                print(f"  ✓ Found: {cmd}")
                print(f"    {version}")
                # Prefer python3 if both exist
                if cmd == 'python3' or python_cmd is None:
                    python_cmd = cmd
        else:
            print(f"  ✗ Not found: {cmd}")
    
    if not python_cmd:
        print("\n❌ ERROR: No Python command found!")
        print("Please install Python 3.10+ from https://www.python.org/")
        sys.exit(1)
    
    # Check if it's Python 3
    version_info = sys.version_info
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 10):
        print(f"\n⚠️  WARNING: Python {version_info.major}.{version_info.minor} detected.")
        print("Python 3.10+ is recommended.")
    
    # Check pip commands
    pip_cmds = ['pip3', 'pip']
    pip_cmd = None
    
    print("\n[2] Checking pip commands...")
    for cmd in pip_cmds:
        if check_command(cmd):
            try:
                result = subprocess.run([cmd, '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print(f"  ✓ Found: {cmd}")
                    print(f"    {result.stdout.strip()}")
                    # Prefer pip3 if both exist
                    if cmd == 'pip3' or pip_cmd is None:
                        pip_cmd = cmd
            except:
                pass
        else:
            print(f"  ✗ Not found: {cmd}")
    
    if not pip_cmd:
        print("\n⚠️  WARNING: No pip command found!")
        print("You may need to install pip or use: python -m ensurepip")
        pip_cmd = f"{python_cmd} -m pip"
        print(f"Using fallback: {pip_cmd}")
    
    # Check venv
    venv_cmd = f"{python_cmd} -m venv"
    print("\n[3] Checking venv...")
    try:
        result = subprocess.run([python_cmd, '-m', 'venv', '--help'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"  ✓ venv module available")
        else:
            print(f"  ✗ venv module not available")
    except:
        print(f"  ✗ Could not check venv")
    
    print("\n" + "=" * 60)
    print("RECOMMENDED COMMANDS FOR THIS SYSTEM:")
    print("=" * 60)
    print(f"\nPython:  {python_cmd}")
    print(f"pip:     {pip_cmd}")
    print(f"venv:    {python_cmd} -m venv")
    
    print("\n" + "=" * 60)
    print("SETUP COMMANDS (copy-paste these):")
    print("=" * 60)
    print(f"""
{python_cmd} -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
{pip_cmd} install -r requirements.txt
{python_cmd} download_dataset.py
""")
    
    print("\n" + "=" * 60)
    print("CLI COMMANDS (copy-paste these):")
    print("=" * 60)
    print(f"""
{python_cmd} cli.py build --mode=memory --csv data/
{python_cmd} cli.py search --mode=memory --model=ltc.ltc --topk=5 --query "space adventure"
{python_cmd} test_cli.py
""")
    
    print("=" * 60)
    
    return python_cmd, pip_cmd

if __name__ == '__main__':
    main()

