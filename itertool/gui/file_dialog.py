"""Native open/save file and folder dialogs for the Streamlit GUI.

Each dialog runs tkinter in a short-lived child process. Streamlit executes the
script on a different thread at every rerun, and a Tk interpreter touched or
garbage-collected from a thread other than its creator fails with "main thread
is not in main loop"; a separate process never shares Tcl with those threads.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_DIALOG_SCRIPT = """
import json, sys, tkinter as tk
from tkinter import filedialog
args = json.loads(sys.argv[1])
root = tk.Tk()
root.withdraw()
root.wm_attributes("-topmost", 1)
filetypes = [tuple(t) for t in args["filetypes"]]
if args["directory"]:
    selected = filedialog.askdirectory(initialdir=args["initial_dir"])
elif args["save"]:
    selected = filedialog.asksaveasfilename(
        initialdir=args["initial_dir"], filetypes=filetypes, defaultextension=args["default_ext"]
    )
else:
    selected = filedialog.askopenfilename(initialdir=args["initial_dir"], filetypes=filetypes)
root.destroy()
sys.stdout.write(selected or "")
"""


def ask_path(
    current: str,
    filetypes: list[tuple[str, str]],
    save: bool = False,
    default_ext: str = "",
    directory: bool = False,
) -> str | None:
    """Show a native file dialog and return the chosen path, or None if cancelled.

    Args:
        current: The path currently in the field; its folder opens first.
        filetypes: ``(label, pattern)`` pairs, e.g. ``("Markdown Files", "*.md")``.
        save: Show a save-as dialog instead of an open dialog.
        default_ext: Extension the save-as dialog appends when none is typed.
        directory: Show a folder picker instead; filetypes, save and
            default_ext are then ignored.

    Returns:
        The chosen path, relative to the working directory when it lies inside
        it, as a POSIX string; None when the dialog was cancelled or failed.
    """
    # An absolute path replaces the working directory; a relative one joins it.
    initial_dir = Path.cwd() / Path(current).parent if current else Path.cwd()
    args = {
        "initial_dir": str(initial_dir),
        "filetypes": filetypes,
        "save": save,
        "default_ext": default_ext,
        "directory": directory,
    }
    try:
        result = subprocess.run(
            [sys.executable, "-c", _DIALOG_SCRIPT, json.dumps(args)],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    selected = result.stdout.strip()
    if not selected:
        return None
    try:
        return Path(selected).relative_to(Path.cwd()).as_posix()
    except ValueError:
        return Path(selected).as_posix()
