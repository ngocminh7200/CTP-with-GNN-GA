"""
TimetableData: a single object holding all the cleaned DataFrames.
Provides load/save and a few convenience query methods.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from config import PROCESSED_DIR


@dataclass
class TimetableData:
    """All cleaned data for the scheduling problem."""
    units: pd.DataFrame
    rooms: pd.DataFrame
    staff: pd.DataFrame
    depts: pd.DataFrame
    programs: pd.DataFrame      # NEW
    timeslots: pd.DataFrame
    schedule: pd.DataFrame
    staff_links: pd.DataFrame

    _FILES = {
        "units":       "units_df.pkl",
        "rooms":       "rooms_df.pkl",
        "staff":       "staff_df.pkl",
        "depts":       "depts_df.pkl",
        "programs":    "programs_df.pkl",        # NEW
        "timeslots":   "timeslots_df.pkl",
        "schedule":    "current_schedule_df.pkl",
        "staff_links": "staff_links_df.pkl",
    }

    @classmethod
    def load(cls, processed_dir: Path = PROCESSED_DIR) -> "TimetableData":
        kwargs = {
            name: pd.read_pickle(processed_dir / fname)
            for name, fname in cls._FILES.items()
        }
        return cls(**kwargs)

    def save(self, processed_dir: Path = PROCESSED_DIR) -> None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        for name, fname in self._FILES.items():
            getattr(self, name).to_pickle(processed_dir / fname)

    # ---------- Sizes ----------
    @property
    def n_units(self) -> int:        return len(self.units)
    @property
    def n_rooms(self) -> int:        return len(self.rooms)
    @property
    def n_staff(self) -> int:        return len(self.staff)
    @property
    def n_depts(self) -> int:        return len(self.depts)
    @property
    def n_programs(self) -> int:     return len(self.programs)        # NEW
    @property
    def n_timeslots(self) -> int:    return len(self.timeslots)

    # ---------- Convenience queries ----------
    def units_in_department(self, dept_code: str) -> pd.DataFrame:
        return self.units[self.units["dept_code"] == dept_code]

    def units_in_program(self, program_id: str) -> pd.DataFrame:        # NEW
        return self.units[self.units["program"] == program_id]

    def rooms_with_capacity(self, min_capacity: int) -> pd.DataFrame:
        return self.rooms[self.rooms["capacity"] >= min_capacity]

    def units_taught_by(self, staff_id: str) -> pd.DataFrame:
        unit_ids = self.staff_links[self.staff_links["staff_id"] == staff_id]["unit_id"]
        return self.units[self.units["unit_id"].isin(unit_ids)]

    def staff_for_unit(self, unit_id: str) -> list[str]:
        return self.staff_links[self.staff_links["unit_id"] == unit_id]["staff_id"].tolist()

    def sample_only(self) -> "TimetableData":
        """Return a copy filtered to the 'regular 10-week' Sample==True units."""
        keep_ids = set(self.units[self.units["Sample"]]["unit_id"])
        u = self.units[self.units["unit_id"].isin(keep_ids)].copy()
        sl = self.staff_links[self.staff_links["unit_id"].isin(keep_ids)].copy()
        sched = self.schedule[self.schedule["unit_id"].isin(keep_ids)].copy()
        rooms_used = set(sched["room_id"].astype(str))
        r = self.rooms[self.rooms["room_id"].astype(str).isin(rooms_used)].copy()
        st = sl.groupby("staff_id").size().reset_index(name="num_units")
        d = u["dept_code"].value_counts().reset_index()
        d.columns = ["dept_code", "num_units"]
        # NEW: rebuild programs_df from the filtered units
        p = (
            u.groupby("program")
            .agg(dept_code=("dept_code", "first"),
                 course_level=("course_level", "first"),
                 num_units=("unit_id", "count"),
                 total_students=("num_students", "sum"))
            .reset_index()
        )
        return TimetableData(
            units=u, rooms=r, staff=st, depts=d, programs=p,
            timeslots=self.timeslots, schedule=sched, staff_links=sl,
        )

    def summary(self) -> str:
        return (
            f"TimetableData(\n"
            f"  units      = {self.n_units}\n"
            f"  rooms      = {self.n_rooms}\n"
            f"  staff      = {self.n_staff}\n"
            f"  depts      = {self.n_depts}\n"
            f"  programs   = {self.n_programs}\n"
            f"  timeslots  = {self.n_timeslots}\n"
            f"  staff_links rows = {len(self.staff_links)}\n"
            f")"
        )

    def __repr__(self) -> str:
        return self.summary()