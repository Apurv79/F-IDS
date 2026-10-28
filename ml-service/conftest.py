import sys
from pathlib import Path

# Ensure root ml-service directory takes precedence in sys.path
root_dir = str(Path(__file__).resolve().parent)
if root_dir not in sys.path or sys.path[0] != root_dir:
    sys.path.insert(0, root_dir)
