import sys
from pathlib import Path

import nltk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
nltk.data.path.insert(0, str(ROOT / "data" / "nltk_data"))
