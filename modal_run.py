"""
Run train.py on a Modal H100 GPU.

Usage: modal run modal_run.py

This is a drop-in replacement for `uv run train.py` that executes
on remote GPU. The agent uses this in the experiment loop instead
of running locally.
"""

import subprocess
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "pandas", "scipy", "scikit-learn")
    .add_local_file("train.py", "/root/train.py")
    .add_local_file("prepare.py", "/root/prepare.py")
    .add_local_dir("data", "/root/data")
)

app = modal.App("autoresearch-gdpa1", image=image)


@app.function(gpu="H100", timeout=4500)  # 75 min timeout (1hr budget + overhead)
def run_training() -> str:
    """Run train.py on H100 and return stdout+stderr."""
    result = subprocess.run(
        ["python", "/root/train.py"],
        capture_output=True,
        text=True,
        cwd="/root",
    )
    output = result.stdout
    if result.stderr:
        output += "\n--- STDERR ---\n" + result.stderr
    if result.returncode != 0:
        output += f"\n--- EXIT CODE: {result.returncode} ---\n"
    return output


@app.local_entrypoint()
def main():
    output = run_training.remote()
    print(output)
