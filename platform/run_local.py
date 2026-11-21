"""Launch the API and one independent job worker; stop both process groups on exit."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


def main():
    stop = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_:stop.set())
    processes = []
    root = Path(__file__).parent
    env = dict(os.environ)
    env.setdefault("DOMAIN_CLASSIFIER", "qwen_qlora")
    env.setdefault("INCIDENT_ANALYSIS_MODE", "ollama")
    env.setdefault("LOCAL_QWEN_MODEL", "qwen3:1.7b")
    env["OLLAMA_NO_CLOUD"] = "1"
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    env["OLLAMA_MAX_LOADED_MODELS"] = "1"
    env["OLLAMA_NUM_PARALLEL"] = "1"
    env.setdefault("SUPPORT_PYTHON",str(root/".venv-support"/"bin"/"python"))
    commands = [[sys.executable,"-m","uvicorn","api:api","--host","127.0.0.1",
                 "--port",os.getenv("PORT","8765")],
                [sys.executable,str(root/"job_worker.py")]]
    try:
        if env["INCIDENT_ANALYSIS_MODE"] == "ollama":
            import urllib.request
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open("http://127.0.0.1:11434/api/tags", timeout=2):
                    pass
            except OSError:
                runtime = Path.home()/"Library"/"Application Support"/"EnterpriseRAG"/"ollama"/"ollama"
                if not runtime.is_file():
                    raise RuntimeError("Local Ollama runtime is missing. No model is downloaded automatically.")
                processes.append(subprocess.Popen([str(runtime),"serve"],cwd=root,env=env,start_new_session=True))
                for _ in range(40):
                    if stop.wait(.25):
                        return
                    try:
                        with opener.open("http://127.0.0.1:11434/api/tags", timeout=1):
                            break
                    except OSError:
                        continue
                else:
                    raise RuntimeError("Local Ollama server did not become ready.")
        for command in commands:
            processes.append(subprocess.Popen(command,cwd=root,env=env,start_new_session=True))
        while not stop.wait(.25):
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("API or job worker exited. Restart the local demo.")
    finally:
        # Groups also include each worker's support subprocess.
        for process in processes:
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


if __name__ == "__main__":
    main()
