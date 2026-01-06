
import os
import subprocess

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Only Run on macOS
if __name__ == "__main__":
    subprocess.run(["streamlit", "run", "ui_v4.py"])



