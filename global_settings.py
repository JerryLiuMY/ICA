from pathlib import Path
import os

# define directories
DESKTOP_PATH = str(Path(os.getcwd()).parent.absolute())
if DESKTOP_PATH == "/Users/mingyu/Desktop":
    DRIVE_PATH = "/Volumes/Sumsung_1T/Projects/ICA"
    DATA_PATH = os.path.join(DRIVE_PATH, "data")
else:
    raise ValueError("Invalid path")
