import os
import subprocess
import tempfile
import shutil
import traceback
import logging
from typing import Tuple, Optional
from shutil import which

# Configure logging
logger = logging.getLogger("ScriptExecutor")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(logging.DEBUG)

class ScriptExecutor:
    """Executes Python scripts safely in the bot's virtualenv with matplotlib figure capture."""

    def __init__(self, timeout: int = 10, py_path: Optional[str] = None):
        """
        Args:
            timeout: max seconds for script execution
            py_path: path to python interpreter (use venv python if None)
        """
        self.timeout = timeout
        self.py_path = py_path or os.path.join(os.getcwd(), "venv", "bin", "python3")
        if not os.path.exists(self.py_path):
            raise FileNotFoundError(f"Python interpreter not found at {self.py_path}")

        self.output_dir = tempfile.mkdtemp(prefix="discord_bot_scripts_")
        os.chmod(self.output_dir, 0o777)  # allow write access
        logger.info(f"Created sandbox directory: {self.output_dir}")

        self.prlimit_bin = which("prlimit")
        if not self.prlimit_bin:
            logger.warning("prlimit not found; CPU/memory/file limits will not be enforced.")

    def _wrap_code(self, code: str) -> str:
        """Wrap user code to capture matplotlib figures."""
        return f"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir('{self.output_dir}')

# --- User code starts here ---
{code}
# --- User code ends here ---

# Save matplotlib figures
for fig_num in plt.get_fignums():
    fig = plt.figure(fig_num)
    fig.savefig(f'output_{{fig_num}}.png', dpi=150, bbox_inches='tight')
plt.close('all')
""".lstrip()

    def execute_script(self, code: str) -> Tuple[bool, str, Optional[str]]:
        """Execute user code safely, clearing old files before running."""
        # Clear all previous files in the sandbox
        for f in os.listdir(self.output_dir):
            path = os.path.join(self.output_dir, f)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                logger.warning(f"Failed to delete {path}: {e}")

        script_path = os.path.join(self.output_dir, "script.py")
        with open(script_path, "w") as f:
            f.write(self._wrap_code(code))
        logger.debug(f"Script written to {script_path}")

        cmd = [self.py_path, script_path]
        if self.prlimit_bin:
            cmd = [
                self.prlimit_bin,
                "--cpu=10",         # max 10 seconds CPU
                "--as=256000000",   # max ~256MB memory
                "--nofile=64",      # max 64 open files
                "--"
            ] + cmd

        env = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONPATH": "",
        }

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.output_dir,
                env=env,
            )
            logger.info("Script execution finished.")

            # Only consider files currently in the sandbox
            image_files = sorted(f for f in os.listdir(self.output_dir) if f.startswith("output_") and f.endswith(".png"))
            image_path = os.path.join(self.output_dir, image_files[0]) if image_files else None
            if image_path:
                logger.info(f"Image generated: {image_path}")

            if result.returncode == 0:
                if result.stdout.strip():
                    logger.debug("Script stdout:\n" + result.stdout.strip())
                return True, result.stdout.strip(), image_path
            else:
                stderr = result.stderr.strip()
                logger.warning(f"Script failed:\n{stderr}")
                return False, f"Error executing script:\n{stderr}", None

        except subprocess.TimeoutExpired:
            logger.error(f"Script timed out after {self.timeout} seconds")
            return False, f"Script timed out after {self.timeout} seconds", None
        except Exception as e:
            logger.exception("Executor error")
            return False, f"Executor error: {e}\n{traceback.format_exc()}", None

    def cleanup(self):
        if os.path.exists(self.output_dir):
            logger.info(f"Cleaning up sandbox directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
