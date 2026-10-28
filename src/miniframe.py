"""
src/miniframe.py
────────────────────────────────────────────────────────────
Drop-in replacement for the tiny subset of pandas DataFrame
operations used in this project.  No compiled extensions —
works even under strict WDAC / AppLocker policies.

Supported API surface:
    MiniFrame(rows)              list[dict] → frame
    MiniFrame.from_csv(path)     read CSV file
    MiniFrame.to_csv(path)       write CSV file
    frame[col]                   returns list of values
    frame[col] == val            returns bool list
    frame[mask]                  boolean-index filter
    frame.empty                  True if no rows
    frame.columns                list of column names
    len(frame)                   row count
    frame.iterrows()             (i, row_dict) pairs
    frame["col"].sum()           numeric sum
    frame["col"].value_counts()  {val: count} dict
    frame["col"].nunique()       unique count
    frame["col"].min/max/mean()  aggregates
    frame.assign(**kwargs)       add/replace columns
    frame.dropna(subset)         drop rows with None/NaN
    frame[mask].reset_index()    filtered frame (reset_index is a no-op)
    pd.DataFrame(rows)           alias: MiniFrame(rows)
    pd.read_csv(path)            alias: MiniFrame.from_csv(path)
    pd.concat(frames)            concatenate list of frames
"""
from __future__ import annotations
import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


class _Column:
    """Represents a single column slice — supports element-wise ops."""

    def __init__(self, values: List[Any], name: str = ""):
        self._v = values
        self.name = name

    # ── Comparisons ─────────────────────────
    def __eq__(self, other):            return _Column([v == other for v in self._v])
    def __ne__(self, other):            return _Column([v != other for v in self._v])
    def __gt__(self, other):            return _Column([_num(v) > other for v in self._v])
    def __lt__(self, other):            return _Column([_num(v) < other for v in self._v])
    def __ge__(self, other):            return _Column([_num(v) >= other for v in self._v])
    def __le__(self, other):            return _Column([_num(v) <= other for v in self._v])

    # Boolean sum — counts True values
    def sum(self):
        # Works for both bool columns and numeric columns
        total = 0
        for v in self._v:
            if v is True or v is False:
                total += int(v)
            elif _is_num(v):
                total += _num(v)
        return total

    # Logical operators for combining boolean masks
    def __and__(self, other):
        o = list(other) if not isinstance(other, _Column) else other._v
        return _Column([bool(a) and bool(b) for a, b in zip(self._v, o)])

    def __or__(self, other):
        o = list(other) if not isinstance(other, _Column) else other._v
        return _Column([bool(a) or bool(b) for a, b in zip(self._v, o)])

    def __invert__(self):
        return _Column([not bool(v) for v in self._v])

    def mean(self):
        nums = [_num(v) for v in self._v if _is_num(v)]
        return sum(nums) / len(nums) if nums else float("nan")

    def min(self):
        nums = [_num(v) for v in self._v if _is_num(v)]
        return min(nums) if nums else None

    def max(self):
        nums = [_num(v) for v in self._v if _is_num(v)]
        return max(nums) if nums else None

    def nunique(self):
        return len(set(v for v in self._v if v is not None))

    def value_counts(self) -> Dict[Any, int]:
        out: Dict[Any, int] = {}
        for v in self._v:
            out[v] = out.get(v, 0) + 1
        return dict(sorted(out.items(), key=lambda x: -x[1]))

    def unique(self) -> List[Any]:
        seen, res = set(), []
        for v in self._v:
            if v not in seen:
                seen.add(v)
                res.append(v)
        return res

    def tolist(self) -> List[Any]:
        return list(self._v)

    def __iter__(self):
        return iter(self._v)

    def __len__(self):
        return len(self._v)

    def __repr__(self):
        return f"_Column({self._v[:5]}{'...' if len(self._v) > 5 else ''})"


# ── helpers ─────────────────────────────────────────────────
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")

def _is_num(v):
    if v is None:
        return False
    try:
        f = float(v)
        return not math.isnan(f)
    except (TypeError, ValueError):
        return False

def _and_masks(*masks):
    result = masks[0]
    for m in masks[1:]:
        result = [a and b for a, b in zip(result, m)]
    return result


class MiniFrame:
    """Minimal pandas-DataFrame-compatible class (pure Python)."""

    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None):
        self._rows: List[Dict[str, Any]] = list(rows) if rows else []

    # ── Constructors ─────────────────────────────────────────
    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "MiniFrame":
        path = str(path)
        if not os.path.exists(path):
            return cls()
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Auto-convert numeric strings
        out = []
        for row in rows:
            out.append({k: _auto(v) for k, v in row.items()})
        return cls(out)

    def to_csv(self, path: Union[str, Path], index: bool = True):
        path = str(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not self._rows:
            Path(path).write_text("", encoding="utf-8")
            return
        cols = list(self._rows[0].keys())
        if index:
            cols = ["index"] + cols
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for i, row in enumerate(self._rows):
                r = dict(row)
                if index:
                    r["index"] = i
                writer.writerow(r)

    # ── Properties ───────────────────────────────────────────
    @property
    def empty(self) -> bool:
        return len(self._rows) == 0

    @property
    def columns(self) -> List[str]:
        return list(self._rows[0].keys()) if self._rows else []

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self._rows), len(self.columns))

    def __len__(self) -> int:
        return len(self._rows)

    def __bool__(self) -> bool:
        return len(self._rows) > 0

    # ── Column access ─────────────────────────────────────────
    def __getitem__(self, key):
        if isinstance(key, str):
            return _Column([r.get(key) for r in self._rows], name=key)
        if isinstance(key, _Column):
            # _Column boolean mask — treat True/truthy as keep
            return MiniFrame([r for r, m in zip(self._rows, key._v) if m])
        if isinstance(key, list):
            # Could be boolean mask (list of bool) or column selector (list of str)
            if key and isinstance(key[0], str):
                return MiniFrame([{k: r.get(k) for k in key} for r in self._rows])
            # boolean mask
            return MiniFrame([r for r, m in zip(self._rows, key) if m])
        if hasattr(key, "__iter__") and not isinstance(key, str):
            mask = list(key)
            return MiniFrame([r for r, m in zip(self._rows, mask) if m])
        raise KeyError(key)

    def __setitem__(self, key: str, value):
        vals = list(value) if hasattr(value, "__iter__") and not isinstance(value, str) else [value] * len(self._rows)
        for i, row in enumerate(self._rows):
            row[key] = vals[i] if i < len(vals) else None

    # ── Row iteration ─────────────────────────────────────────
    def iterrows(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        for i, row in enumerate(self._rows):
            yield i, row

    def itertuples(self, index=True, name="Row"):
        from collections import namedtuple
        cols = self.columns
        NT = namedtuple(name, (["Index"] if index else []) + cols)
        for i, row in enumerate(self._rows):
            vals = [row.get(c) for c in cols]
            yield NT(*([i] if index else []), *vals)

    # ── Filtering helpers ─────────────────────────────────────
    def reset_index(self, drop=True):
        return MiniFrame(list(self._rows))

    def dropna(self, subset=None):
        def _ok(row):
            cols = subset if subset else row.keys()
            return all(row.get(c) is not None and row.get(c) != "" for c in cols)
        return MiniFrame([r for r in self._rows if _ok(r)])

    def drop_duplicates(self, subset=None):
        seen, out = set(), []
        for row in self._rows:
            key = tuple((k, row.get(k)) for k in (subset or self.columns))
            if key not in seen:
                seen.add(key)
                out.append(row)
        return MiniFrame(out)

    def sort_values(self, by, ascending=True):
        col = by if isinstance(by, str) else by[0]
        return MiniFrame(sorted(self._rows, key=lambda r: (_num(r.get(col, 0)),), reverse=not ascending))

    # ── Column operations ─────────────────────────────────────
    def assign(self, **kwargs) -> "MiniFrame":
        new_rows = [dict(r) for r in self._rows]
        for col, vals in kwargs.items():
            if callable(vals):
                computed = vals(MiniFrame(new_rows))
                vals = list(computed) if hasattr(computed, "__iter__") else [computed] * len(new_rows)
            elif hasattr(vals, "__iter__") and not isinstance(vals, str):
                vals = list(vals)
            else:
                vals = [vals] * len(new_rows)
            for i, row in enumerate(new_rows):
                row[col] = vals[i] if i < len(vals) else None
        return MiniFrame(new_rows)

    def groupby(self, col):
        groups: Dict[Any, List[Dict]] = {}
        for row in self._rows:
            key = row.get(col)
            groups.setdefault(key, []).append(row)
        return _GroupBy(groups, col)

    def sample(self, n=None, frac=None, random_state=None):
        import random
        rng = random.Random(random_state)
        rows = list(self._rows)
        rng.shuffle(rows)
        if frac is not None:
            n = int(len(rows) * frac)
        return MiniFrame(rows[:n])

    def head(self, n=5):
        return MiniFrame(self._rows[:n])

    def tail(self, n=5):
        return MiniFrame(self._rows[-n:])

    def copy(self):
        return MiniFrame([dict(r) for r in self._rows])

    def to_dict(self, orient="records"):
        if orient == "records":
            return list(self._rows)
        if orient == "list":
            return {col: [r.get(col) for r in self._rows] for col in self.columns}
        return list(self._rows)

    def __repr__(self):
        return f"MiniFrame({len(self._rows)} rows × {len(self.columns)} cols)"


class _Series:
    """
    Lightweight pandas Series replacement.
    Wraps an ordered dict of {key: value} from groupby/value_counts results.
    Supports: .index, .values, .sort_values(), .items(), len(), iteration.
    """
    def __init__(self, data: Dict[Any, Any]):
        self._data = dict(data)

    @property
    def index(self) -> List[Any]:
        return list(self._data.keys())

    @property
    def values(self) -> List[Any]:
        return list(self._data.values())

    def sort_values(self, ascending: bool = True) -> "_Series":
        sorted_items = sorted(self._data.items(), key=lambda x: x[1], reverse=not ascending)
        return _Series(dict(sorted_items))

    def sort_index(self, ascending: bool = True) -> "_Series":
        sorted_items = sorted(self._data.items(), key=lambda x: x[0], reverse=not ascending)
        return _Series(dict(sorted_items))

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __repr__(self):
        return f"_Series({self._data})"


class _GroupBy:
    def __init__(self, groups, col):
        self._groups = groups
        self._col = col

    def __iter__(self):
        for key, rows in self._groups.items():
            yield key, MiniFrame(rows)

    def size(self) -> _Series:
        return _Series({k: len(v) for k, v in self._groups.items()})

    def count(self) -> _Series:
        return self.size()



def _auto(v: str) -> Any:
    """Try to convert a CSV string to int/float, else keep as str."""
    if v == "" or v is None:
        return None
    try:
        i = int(v)
        return i
    except ValueError:
        pass
    try:
        f = float(v)
        return f
    except ValueError:
        pass
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    return v


# ── Public pandas-compatible API ─────────────────────────────

def DataFrame(data=None, columns=None) -> MiniFrame:
    if data is None:
        return MiniFrame()
    if isinstance(data, list):
        if not data:
            return MiniFrame()
        if isinstance(data[0], dict):
            return MiniFrame(data)
        # list of lists/tuples with explicit columns
        if columns:
            return MiniFrame([dict(zip(columns, row)) for row in data])
    if isinstance(data, dict):
        # dict of lists → list of dicts
        cols = list(data.keys())
        length = len(data[cols[0]]) if cols else 0
        rows = [{c: data[c][i] for c in cols} for i in range(length)]
        return MiniFrame(rows)
    return MiniFrame()


def read_csv(path, **kwargs) -> MiniFrame:
    return MiniFrame.from_csv(path)


def concat(frames: List[MiniFrame], ignore_index: bool = True) -> MiniFrame:
    all_rows = []
    for f in frames:
        all_rows.extend(f._rows)
    return MiniFrame(all_rows)


def isna(val) -> bool:
    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False


def notna(val) -> bool:
    return not isna(val)
