import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in (REPO, os.path.join(REPO, "reproduce")):
    if path not in sys.path:
        sys.path.insert(0, path)
