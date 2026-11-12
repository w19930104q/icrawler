import os
import sys

# Make the in-tree package importable regardless of the directory pytest is
# launched from (the project is not installed in the test environment).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
