import os
import subprocess
import tempfile
import shutil
import traceback
import logging
from typing import Tuple, Optional
from shutil import which
import pwd
import grp
import stat

try:
    import resource  # POSIX only
except Exception:  # pragma: no cover
    resource = None

# Configure logging
logger = logging.getLogger("ScriptExecutor")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(logging.DEBUG)

class ScriptExecutor:
    """Executes Python scripts with stronger isolation by dropping privileges to a dedicated sandbox user.

    Notes
    -----
    * Linux-only. Requires either:
        - Running as root (so we can setuid/setgid in the child), OR
        - Passwordless `sudo -n -u <sandbox_user>` allowed for the current user.
    * You should provision a locked-down user beforehand, e.g.:
        sudo useradd --system --create-home --shell /usr/sbin/nologin sandbox
        sudo usermod -L sandbox
      Give it no shell and no extra group memberships.
    * For defense-in-depth, consider also running inside a container or firejail/nsjail.
    """

    def __init__(
        self,
        timeout: int = 10,
        py_path: Optional[str] = None,
        sandbox_user: str = "sandboxuser",
    ):
        self.timeout = timeout
        self.py_path = py_path or os.path.join(os.getcwd(), "venv", "bin", "python3")
        if not os.path.exists(self.py_path):
            raise FileNotFoundError(f"Python interpreter not found at {self.py_path}")

        # Resolve sandbox user and primary group
        try:
            pw = pwd.getpwnam(sandbox_user)
        except KeyError:
            raise RuntimeError(
                f"Sandbox user '{sandbox_user}' does not exist. Create it (see class docstring)."
            )
        self.sandbox_user = sandbox_user
        self.sandbox_uid = pw.pw_uid
        self.sandbox_gid = pw.pw_gid
        self.sandbox_home = pw.pw_dir

        # Create an isolated working directory world-writable (since we can't chown without root)
        self.output_dir = tempfile.mkdtemp(prefix="discord_bot_scripts_")
        os.chmod(self.output_dir, 0o777)

        # Also prepare a tmp subdir
        self.tmp_dir = os.path.join(self.output_dir, "tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.chmod(self.tmp_dir, 0o777)

        logger.info(f"Created sandbox directory: {self.output_dir}")

        self.prlimit_bin = which("prlimit")
        self.sudo_bin = which("sudo")
        if not self.prlimit_bin:
            logger.warning("prlimit not found; CPU/memory/file limits will rely on 'resource' (if available).")

    # ---------------- internal helpers ----------------

    def _wrap_code(self, code: str) -> str:
        """Wrap user code to capture matplotlib figures and force write dir."""
        return f"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Force all reads/writes to the working directory
os.chdir('{self.output_dir}')
os.environ['TMPDIR'] = '{self.tmp_dir}'

# --- User code starts here ---
{code}
# --- User code ends here ---

# Save matplotlib figures
for fig_num in plt.get_fignums():
    fig = plt.figure(fig_num)
    fig.savefig(f'output_{{fig_num}}.png', dpi=150, bbox_inches='tight')
plt.close('all')
""".lstrip()

    def _clear_dir(self, path: str) -> None:
        for f in os.listdir(path):
            p = os.path.join(path, f)
            try:
                if os.path.islink(p) or os.path.isfile(p):
                    os.unlink(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p)
            except Exception as e:
                logger.warning(f"Failed to delete {p}: {e}")

    def _preexec_drop_privs(self):
        if os.geteuid() != 0:
            return None

        def _fn():
            try:
                os.setsid()
                os.umask(0o077)
                try:
                    os.setgroups([])
                except Exception:
                    pass
                os.setgid(self.sandbox_gid)
                os.setuid(self.sandbox_uid)
                if resource is not None:
                    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
                    resource.setrlimit(resource.RLIMIT_AS, (256_000_000, 256_000_000))
                    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
                    if hasattr(resource, 'RLIMIT_NPROC'):
                        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
            except Exception as e:
                os.write(2, f"Privilege drop failed: {e}\n".encode())
                os._exit(1)

        return _fn

    def _build_base_cmd(self, script_path: str):
        cmd = [self.py_path, script_path]
        if self.prlimit_bin:
            cmd = [
                self.prlimit_bin,
                "--cpu=10",
                "--as=256000000",
                "--nofile=64",
                "--nproc=64",
                "--"
            ] + cmd

        if os.geteuid() != 0:
            if not self.sudo_bin:
                logger.warning("Not root and 'sudo' not found—cannot drop user. Child will inherit current UID!")
                return cmd, None
            cmd = [self.sudo_bin, "-n", "-u", self.sandbox_user, "--"] + cmd
            return cmd, None

        return cmd, self._preexec_drop_privs()

    def execute_script(self, code: str) -> Tuple[bool, str, Optional[str]]:
        self._clear_dir(self.output_dir)
        self._clear_dir(self.tmp_dir)

        script_path = os.path.join(self.output_dir, "script.py")
        with open(script_path, "w") as f:
            f.write(self._wrap_code(code))
        os.chmod(script_path, 0o777)
        logger.debug(f"Script written to {script_path}")

        cmd, preexec = self._build_base_cmd(script_path)

        env = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONPATH": "",
            "TMPDIR": self.tmp_dir,
        }

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.output_dir,
                env=env,
                preexec_fn=preexec,
            )
            logger.info("Script execution finished.")

            image_files = sorted(
                f for f in os.listdir(self.output_dir) if f.startswith("output_") and f.endswith(".png")
            )
            image_path = os.path.join(self.output_dir, image_files[0]) if image_files else None
            if image_path:
                logger.info(f"Image generated: {image_path}")

            if result.returncode == 0:
                if result.stdout.strip():
                    logger.debug("Script stdout:\n" + result.stdout.strip())
                return True, result.stdout.strip(), image_path
            else:
                stderr = result.stderr.strip()
                logger.warning(f"Script failed (rc={result.returncode}):\n{stderr}")
                return False, f"Error executing script (rc={result.returncode}):\n{stderr}", None

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
