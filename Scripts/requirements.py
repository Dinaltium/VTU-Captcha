"""
requirements.py
Helper script to ensure conda environment 'tfenv' exists and install Python requirements into it.
Run from project root or directly: python Scripts\requirements.py
"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REQ_FILE = os.path.join(SCRIPT_DIR, "requirements.txt")


def run(cmd, check=True):
    print("$", " ".join(cmd))
    res = subprocess.run(cmd)
    if check and res.returncode != 0:
        raise SystemExit(res.returncode)


def main():
    # Check conda availability
    try:
        subprocess.run(["conda", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("Error: 'conda' not found in PATH. Please install Anaconda or Miniconda and ensure 'conda' is available.")
        sys.exit(1)

    # Check if env exists
    envs = subprocess.check_output(["conda", "env", "list"], universal_newlines=True)
    if "tfenv" not in envs:
        print("Creating conda environment 'tfenv' (python=3.9)")
        run(["conda", "create", "-y", "-n", "tfenv", "python=3.9"])
    else:
        print("Conda environment 'tfenv' already exists")

    # Install requirements into the env using conda run to avoid activation issues in scripts
    if not os.path.exists(REQ_FILE):
        print(f"Error: requirements.txt not found at {REQ_FILE}")
        sys.exit(1)

    print("Installing pip packages into 'tfenv'...")
    run(["conda", "run", "-n", "tfenv", "pip", "install", "-r", REQ_FILE])

    print("All done. To start the backend interactively use:")
    print("  call conda activate tfenv && python backend\\python\\api.py")


if __name__ == '__main__':
    main()
