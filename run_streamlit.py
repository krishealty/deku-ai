"""Streamlit entry point for the traffic intelligence system."""
import subprocess
import sys

if __name__ == "__main__":
    # Run Streamlit app
    subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"])

