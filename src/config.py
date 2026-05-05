"""Central configuration. Edit this file to change scope or paths."""
from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---------- Scope ----------
# Department prefixes (2 letters from Module.Host Key) to keep.
# Use the string "ALL" to keep every department.
DEPARTMENT_PREFIXES = "ALL"   # e.g. ["AC", "FM", "EC"] or "ALL"

# Term: 'AT' (Autumn), 'WT' (Winter), 'ST' (Summer), or None for all.
TERM = "AT"

# ---------- Timeslot grid ----------
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
HOURS = list(range(9, 18))    # 09:00 to 17:00 starts -> 9 hourly slots

# Whether to apply the 'Sample == True' (regular 10-week) filter at preprocess time
APPLY_SAMPLE_FILTER = False