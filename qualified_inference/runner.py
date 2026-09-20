"""Execute a declared route with writes confined to its external run directory."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]

def install_guard(output):
    output = Path(output).resolve()
    allowed = [ROOT, output, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
               Path("/System"), Path("/usr"), Path("/Library/Fonts"),
               Path("/etc").resolve(), Path("/dev"), Path.home() / "Library/Fonts"]
    def audit(event, args):
        if event in ("os.remove", "os.rmdir", "os.mkdir", "os.rename"):
            paths = args[:2] if event == "os.rename" else args[:1]
            for raw in paths:
                path = Path(os.fsdecode(raw)).resolve()
                if path != output and output not in path.parents:
                    raise PermissionError("Filesystem change outside run directory: " + str(path))
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("Network access is not an analysis input")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing:
            if path != Path("/dev/null") and path != output and output not in path.parents:
                raise PermissionError("Write outside the run directory: " + str(path))
        elif not any(path == base or base in path.parents for base in allowed):
            raise PermissionError("Undeclared external input: " + str(path))
    sys.addaudithook(audit)

def main():
    import json
    output = Path(sys.argv[1]).resolve()
    script = sys.argv[2]
    routes = json.loads((ROOT / "WORKFLOW.json").read_text())["routes"]
    if script not in {r["script"] for r in routes}:
        raise ValueError("Undeclared analysis program")
    sys.path.insert(0, str(ROOT))
    install_guard(output)
    sys.argv = [str(ROOT / script), *sys.argv[3:]]
    runpy.run_path(str(ROOT / script), run_name="__main__")

if __name__ == "__main__":
    main()
