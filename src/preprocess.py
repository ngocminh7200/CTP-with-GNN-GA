"""
Stage 1: Preprocessing pipeline.

Run once:    python src/preprocess.py
Output:      data/processed/*.pkl
Subsequent:  TimetableData.load() in any other module.
"""
from __future__ import annotations
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from config import (
    RAW_DIR, PROCESSED_DIR,
    DEPARTMENT_PREFIXES, TERM,
    DAYS, HOURS,
    APPLY_SAMPLE_FILTER,
)
from timetable_data import TimetableData


# ---------- Constants ----------

_STAFF_SPLIT_RE = re.compile(r"\s*[/;&]\s*|\s+&\s+")


# Standard AT week patterns considered "regular" 10-week courses.
_REGULAR_AT_PATTERNS = {
    frozenset(range(1, 11)),         # weeks 1-10
    frozenset(range(2, 12)),         # weeks 2-11
    frozenset(range(1, 12)) - {6},   # weeks 1-11 minus reading week
    frozenset(range(2, 12)) - {6},   # weeks 2-11 minus reading week
    frozenset(range(1, 12)),         # weeks 1-11
}


# ---------- Helpers ----------

def _print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _hour_from_time(value) -> int:
    """Robustly extract the hour from a time / string / datetime cell."""
    try:
        return pd.to_datetime(str(value)).hour
    except Exception:
        return -1


def _normalize_group(value) -> str:
    """Normalize group identifiers to a canonical string form.

        - 'L' / 'l'                       -> 'L'
        - integers / numeric strings      -> zero-padded 2-digit, e.g. 1 -> '01'
        - anything else                   -> uppercased trimmed string
    """
    s = str(value).strip()
    if s.upper() == "L":
        return "L"
    try:
        return f"{int(float(s)):02d}"
    except (ValueError, TypeError):
        return s.upper()




_STAFF_SPLIT_RE = re.compile(r"(?:_X000D_|[/;&\n\r])+", re.IGNORECASE)

def _normalize_staff_string(raw: str) -> list[str]:
    """Split a raw 'Staff' cell into a list of canonical staff identifiers.

    Splits on:
        * '_X000D_'   (Excel's escape for carriage return inside cells)
        * '\\n', '\\r' (literal newlines)
        * '/', ';', '&' (occasional inline separators)

    For each part: collapse whitespace, strip, uppercase.
    Returns an empty list if the cell is null or whitespace.
    """
    if not isinstance(raw, str) or raw.strip() == "":
        return []
    parts = _STAFF_SPLIT_RE.split(raw)
    out = []
    for p in parts:
        canonical = re.sub(r"\s+", " ", p).strip().upper()
        if canonical:
            out.append(canonical)
    return out

def _parse_week_list(s) -> frozenset[int]:
    """Parse '1,2,3,...' into a frozenset of integers."""
    if not isinstance(s, str) or s.strip() == "":
        return frozenset()
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
    return frozenset(out)


# ---------- Loaders ----------

def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _print_header("LOADING RAW FILES")
    groups = pd.read_excel(RAW_DIR / "Group_Sizes.xlsx")
    rooms = pd.read_excel(RAW_DIR / "Teaching_Rooms.xlsx")
    tt = pd.read_excel(RAW_DIR / "TT_by_course.xlsx")
    print(f"Group_Sizes:    {groups.shape}, columns: {list(groups.columns)}")
    print(f"Teaching_Rooms: {rooms.shape}, columns: {list(rooms.columns)}")
    print(f"TT_by_course:   {tt.shape}, columns: {list(tt.columns)}")
    return groups, rooms, tt


# ---------- Filtering ----------

def filter_term(tt: pd.DataFrame, term: str | None) -> pd.DataFrame:
    if term is None:
        return tt
    col = f"{term} Weeks"
    if col not in tt.columns:
        raise ValueError(f"No column '{col}'. Found: {list(tt.columns)}")
    mask = tt[col].notna() & (tt[col].astype(str).str.strip() != "")
    print(f"Term filter '{term}': kept {mask.sum()} of {len(tt)} rows")
    return tt[mask].copy()


def filter_departments(tt: pd.DataFrame, prefixes) -> pd.DataFrame:
    if prefixes == "ALL":
        print("Department filter: ALL (no filter applied)")
        return tt
    if not isinstance(prefixes, (list, tuple)) or len(prefixes) == 0:
        raise ValueError(
            f"DEPARTMENT_PREFIXES must be 'ALL' or a non-empty list, got {prefixes!r}"
        )
    pattern = "|".join(f"^{p}" for p in prefixes)
    mask = tt["Module.Host Key"].astype(str).str.match(pattern)
    print(f"Department filter {prefixes}: kept {mask.sum()} of {len(tt)} rows")
    return tt[mask].copy()


def filter_anomalies(tt: pd.DataFrame) -> pd.DataFrame:
    """Drop:
       - course-level codes 'D' or 'E' (third char of module code)
       - sessions shorter than 30 minutes
    """
    n_before = len(tt)
    level = tt["Module.Host Key"].astype(str).str[2:3]
    mask_level = ~level.isin(["D", "E"])
    n_dropped_level = (~mask_level).sum()

    def dur_minutes(row):
        s = pd.to_datetime(str(row["Start"]), errors="coerce")
        e = pd.to_datetime(str(row["End"]),   errors="coerce")
        if pd.isna(s) or pd.isna(e):
            return -1
        return (e - s).total_seconds() / 60.0
    durations = tt.apply(dur_minutes, axis=1)
    mask_dur = durations >= 30
    n_dropped_dur = (~mask_dur).sum()

    keep = mask_level & mask_dur
    print(f"Anomaly filter: dropped {n_dropped_level} D/E-level rows, "
          f"{n_dropped_dur} short sessions")
    print(f"  kept {keep.sum()} of {n_before} rows")
    return tt[keep].copy()


# ---------- Builders ----------

def build_units_df(
    tt: pd.DataFrame, groups: pd.DataFrame, term: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per (module, group) scheduling unit, with multi-session info.

    Returns (units_df, staff_links_df).
    """
    _print_header("BUILDING UNITS DATAFRAME")

    tt = tt.copy()
    tt["module_family"] = tt["Module.Host Key"].astype(str)
    tt["module_code"] = (
        tt["Description"].astype(str).str.strip().str.split(n=1).str[0]
    )
    tt["dept_code"] = tt["module_family"].str[:2]
    tt["course_level"] = tt["module_family"].str[2:3]
    tt["program"] = tt["dept_code"] + tt["course_level"]
    tt["group_norm"] = tt["Group"].apply(_normalize_group)

    # Lectures use "01" in TT but "L" in Group_Sizes -> normalize
    tt.loc[tt["Type"].astype(str).str.upper() == "LECTURE", "group_norm"] = "L"
    tt["unit_id"] = tt["module_code"] + "__" + tt["group_norm"]

    tt["start_hour"] = tt["Start"].apply(_hour_from_time)
    tt["end_hour"] = tt["End"].apply(_hour_from_time)

    print("Aggregating sessions per unit...")
    day_order = {d: i for i, d in enumerate(DAYS)}
    tt["_day_idx"] = tt["Scheduled Day"].map(day_order).fillna(99).astype(int)

    def session_record(row):
        return {
            "day": row["Scheduled Day"],
            "start_hour": int(row["start_hour"]) if row["start_hour"] >= 0 else None,
            "end_hour":   int(row["end_hour"])   if row["end_hour"]   >= 0 else None,
            "room": row["Location"],
            "staff_raw": row["Staff"],
        }
    tt["_session"] = tt.apply(session_record, axis=1)

    tt = tt.sort_values(["unit_id", "_day_idx", "start_hour"])
    sessions_per_unit = tt.groupby("unit_id", sort=False)["_session"].apply(list)

    units = tt.drop_duplicates(subset="unit_id", keep="first").copy()
    units["sessions"] = units["unit_id"].map(sessions_per_unit)
    units["n_sessions"] = units["sessions"].apply(len)
    print(f"Unit count: {len(units)} | total sessions: {units['n_sessions'].sum()} | "
          f"avg sessions/unit: {units['n_sessions'].mean():.2f}")

    # Primary-session attributes (used for chromosome / fitness / ground truth)
    units["current_day"] = units["Scheduled Day"]
    units["current_start_hour"] = units["start_hour"]
    units["duration_hours"] = (units["end_hour"] - units["start_hour"]).clip(lower=1)
    units["current_room"] = units["Location"]

    # ---- Student-count lookup (uses normalized keys on both sides) ----
    g = groups.copy()
    g["group_norm"] = g["Group No."].apply(_normalize_group)
    g["Code"] = g["Code"].astype(str)
    lookup = g.set_index(["Code", "group_norm"])["NUMSTU"].to_dict()

    def student_count(row):
        if str(row["Type"]).upper() == "LECTURE":
            key = (row["module_code"], "L")
            if key in lookup:
                return lookup[key]
        key = (row["module_code"], row["group_norm"])
        return lookup.get(key, np.nan)

    units["num_students"] = units.apply(student_count, axis=1)
    n_missing = units["num_students"].isna().sum()
    if n_missing:
        med = units["num_students"].median()
        print(f"⚠ {n_missing} units missing student counts -> filled with median ({med:.0f})")
        units["num_students"] = units["num_students"].fillna(med)
    else:
        print("✓ All units matched a student count")

    # ---- Sample flag ----
    week_col = f"{term} Weeks"
    units["weeks_set"] = units[week_col].apply(_parse_week_list)
    units["Sample"] = units["weeks_set"].apply(lambda ws: ws in _REGULAR_AT_PATTERNS)
    print(f"'Sample' courses (regular 10-week AT pattern): "
          f"{units['Sample'].sum()} of {len(units)}")

    # ---- Build staff_links (long format) ----
    print("Splitting multi-staff sessions...")
    rows = []
    for _, u in units.iterrows():
        for s in _normalize_staff_string(u.get("Staff", "")):
            rows.append({"unit_id": u["unit_id"], "staff_id": s})
    staff_links = pd.DataFrame(rows, columns=["unit_id", "staff_id"]).drop_duplicates()
    print(f"  staff_links rows: {len(staff_links)} | "
          f"distinct staff: {staff_links['staff_id'].nunique()}")

    # ---- Tidy schema ----
    units = units.rename(columns={
        "Description": "description",
        "Type": "session_type",
        "module_code": "module",
    })
    units = units[[
        "unit_id", "module", "module_family", "group_norm",
        "description", "session_type",
        "dept_code", "course_level", "program",
        "duration_hours", "num_students", "n_sessions", "sessions",
        "current_day", "current_start_hour", "current_room",
        "Sample",
    ]].rename(columns={"group_norm": "group"})

    print(f"Final units_df: {len(units)} rows × {len(units.columns)} cols")
    print(f"Session type distribution:\n{units['session_type'].value_counts()}")
    return units, staff_links


def build_rooms_df(rooms_raw: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    _print_header("BUILDING ROOMS DATAFRAME")
    rooms = rooms_raw.rename(columns={
        "Host Key": "room_id",
        "Capacity": "capacity",
        "Suitability": "suitability",
        "Zone Name": "zone",
        "Name": "room_name",
    }).copy()
    rooms["room_id"] = rooms["room_id"].astype(str)

    used = set(units["current_room"].dropna().astype(str).unique())
    in_file = set(rooms["room_id"].unique())
    missing = used - in_file
    if missing:
        print(f"⚠ {len(missing)} rooms in units but not in Teaching_Rooms.xlsx "
              "(will drop those units):")
        for r in sorted(missing):
            print(f"    {r}")
    rooms = rooms[rooms["room_id"].isin(used)].copy()
    print(f"Final rooms_df: {len(rooms)} rooms")
    if len(rooms):
        print(f"Capacity stats: min={rooms.capacity.min():.0f}, "
              f"med={rooms.capacity.median():.0f}, max={rooms.capacity.max():.0f}")
    return rooms


def build_staff_df(staff_links: pd.DataFrame) -> pd.DataFrame:
    _print_header("BUILDING STAFF DATAFRAME")
    s = staff_links.groupby("staff_id").size().reset_index(name="num_units")
    print(f"Distinct staff (after aliasing): {len(s)}")
    if len(s):
        print(f"Workload stats: min={s.num_units.min()}, "
              f"med={s.num_units.median()}, max={s.num_units.max()}")
    return s


def build_depts_df(units: pd.DataFrame) -> pd.DataFrame:
    _print_header("BUILDING DEPARTMENTS DATAFRAME")
    d = units["dept_code"].value_counts().reset_index()
    d.columns = ["dept_code", "num_units"]
    print(f"Departments in scope: {len(d)}")
    return d

def build_programs_df(units: pd.DataFrame) -> pd.DataFrame:
    """One row per (department, level) program.

    program_id = dept_code + course_level, e.g. 'AC1' = Accounting year-1 UG,
    'EC4' = Economics Masters. Programs are first-class nodes in the heterogeneous
    graph (alongside courses, professors, rooms, departments).
    """
    _print_header("BUILDING PROGRAMS DATAFRAME")
    p = (
        units.groupby("program")
        .agg(
            dept_code=("dept_code", "first"),
            course_level=("course_level", "first"),
            num_units=("unit_id", "count"),
            total_students=("num_students", "sum"),
        )
        .reset_index()
    )
    print(f"Programs in scope: {len(p)}")
    print("Top programs by unit count:")
    print(p.sort_values("num_units", ascending=False).head(15).to_string(index=False))
    return p

def build_timeslots_df() -> pd.DataFrame:
    rows = [(d, h, f"{d}_{h:02d}") for d in DAYS for h in HOURS]
    return pd.DataFrame(rows, columns=["day", "start_hour", "timeslot_id"])


def drop_invalid_units(
    units: pd.DataFrame, rooms: pd.DataFrame, timeslots: pd.DataFrame
) -> pd.DataFrame:
    _print_header("DROPPING UNITS WITH INVALID ROOM OR TIMESLOT")
    valid_rooms = set(rooms["room_id"].astype(str))
    valid_days = set(timeslots["day"])
    valid_hours = set(timeslots["start_hour"])

    n_before = len(units)
    bad_room = ~units["current_room"].astype(str).isin(valid_rooms)
    bad_day = ~units["current_day"].isin(valid_days)
    bad_hour = ~units["current_start_hour"].isin(valid_hours)

    print("Drop reasons (units may be flagged by more than one):")
    print(f"  bad room: {bad_room.sum()}")
    print(f"  bad day:  {bad_day.sum()}")
    print(f"  bad hour: {bad_hour.sum()}")

    keep = ~(bad_room | bad_day | bad_hour)
    units_clean = units[keep].copy()
    print(f"Kept {len(units_clean)} of {n_before} units")
    return units_clean


def build_current_schedule_df(
    units: pd.DataFrame, timeslots: pd.DataFrame, rooms: pd.DataFrame
) -> pd.DataFrame:
    _print_header("BUILDING CURRENT SCHEDULE (ground truth)")
    sched = units[["unit_id", "current_day", "current_start_hour", "current_room"]].copy()
    sched["timeslot_id"] = (
        sched["current_day"].astype(str) + "_" +
        sched["current_start_hour"].astype(int).map(lambda h: f"{h:02d}")
    )
    sched = sched.rename(columns={"current_room": "room_id"})
    sched = sched[["unit_id", "timeslot_id", "room_id"]]

    valid_ts = set(timeslots["timeslot_id"])
    valid_rm = set(rooms["room_id"].astype(str))
    bad_ts = (~sched["timeslot_id"].isin(valid_ts)).sum()
    bad_rm = (~sched["room_id"].astype(str).isin(valid_rm)).sum()
    if bad_ts or bad_rm:
        print(f"⚠ Still {bad_ts} bad timeslots / {bad_rm} bad rooms — investigate.")
    else:
        print("✓ All schedule entries are valid")
    return sched


# ---------- Main ----------

def main() -> TimetableData:
    groups, rooms_raw, tt_raw = load_raw()

    tt = filter_term(tt_raw, TERM)
    tt = filter_departments(tt, DEPARTMENT_PREFIXES)
    tt = filter_anomalies(tt)

    units, staff_links = build_units_df(tt, groups, TERM)
    rooms = build_rooms_df(rooms_raw, units)
    timeslots = build_timeslots_df()

    units = drop_invalid_units(units, rooms, timeslots)
    staff_links = staff_links[staff_links["unit_id"].isin(units["unit_id"])].copy()

    if APPLY_SAMPLE_FILTER:
        n_before = len(units)
        units = units[units["Sample"]].copy()
        staff_links = staff_links[staff_links["unit_id"].isin(units["unit_id"])].copy()
        print(f"\nSample filter applied: kept {len(units)} of {n_before} units")

    staff = build_staff_df(staff_links)
    depts = build_depts_df(units)
    programs = build_programs_df(units)  # NEW
    schedule = build_current_schedule_df(units, timeslots, rooms)

    data = TimetableData(
        units=units, rooms=rooms, staff=staff,
        depts=depts, programs=programs,  # NEW
        timeslots=timeslots, schedule=schedule,
        staff_links=staff_links,
    )
    data.save()

    _print_header("SUMMARY")
    print(data.summary())
    return data


if __name__ == "__main__":
    main()