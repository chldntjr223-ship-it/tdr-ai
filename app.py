import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

app_script = os.path.join(ROOT_DIR, "web", "streamlit_app.py")

# If executed via streamlit run app.py (e.g. Hugging Face Spaces / Streamlit Cloud)
if "streamlit" in sys.modules or os.environ.get("STREAMLIT_SERVER_PORT") or "--server.port" in sys.argv:
    with open(app_script, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, app_script, "exec"), globals())
elif __name__ == "__main__":
    # If run directly with python app.py
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_script, "--server.port", "7860", "--server.headless", "true"])
