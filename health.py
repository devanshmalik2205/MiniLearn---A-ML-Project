"""Checks a CSV before modelling and describes it in plain language.
Each suggestion is {level: bad|warn|info, title, detail}: detail says what is wrong and what to do."""
import re

import numpy as np

import minilearn as ml

ODD_MISSING = {"n/a", "n.a.", "-", "--", "unknown", "missing", "not available", "undefined", "#n/a"}


def _clean_number(v):
    return re.sub(r"[$€£%,\s]", "", v)


def check(header, rows, info, notes=()):
    out = list(notes)
    n = len(rows)
    add = lambda level, title, detail: out.append(dict(level=level, title=title, detail=detail))

    if n < 30:
        add("bad", f"Only {n} rows", "Predicting needs at least 30 rows with a target value to compare models fairly. Add more rows if you can.")
    elif n < 100:
        add("warn", f"Small dataset ({n} rows)", "Scores will be noisy with so few rows. Treat the ranking as a rough guide, and look at the ± column.")

    blanks = [i + 1 for i, h in enumerate(header) if not h.strip()]
    if blanks:
        add("bad", "Some column names are empty", f"Give every column a name (column numbers {blanks[:5]}).")
    dup_names = sorted({h for h in header if header.count(h) > 1})
    if dup_names:
        add("bad", "Repeated column names", f"{', '.join(dup_names[:4])} appear more than once, so only the first is used. Rename them so each is unique.")

    dupes = n - len({tuple(r) for r in rows})
    if dupes:
        add("warn", f"{dupes} duplicate rows", "Identical rows are counted twice and can make scores look better than they are. Remove the duplicates.")

    for i, c in enumerate(info):
        name, vals = c["name"], [r[i] for r in rows]
        present = [v for v in vals if not ml.is_missing(v)]
        pct = 100 * c["missing"] / n
        if c["type"] == "empty":
            add("warn", f"'{name}' is completely empty", "Delete this column.")
            continue
        if pct > 60:
            add("warn", f"'{name}' is {pct:.0f}% empty", "Too little data to learn from. Delete the column, or fill it in properly.")
        elif pct > 5:
            how = "MiniLearn fills these with the column's median." if c["type"] == "numeric" else "Blank text cells count as having no category."
            add("info", f"'{name}' has {pct:.0f}% gaps", f"{how} If you know the real values, fill them in.")
        if c["unique"] == 1:
            add("warn", f"'{name}' never changes", "A column with a single value tells a model nothing. Delete it.")
        if c["type"] == "numeric" and present and ml.looks_like_id(present):
            add("info", f"'{name}' looks like an ID", "Row numbers carry no meaning, so MiniLearn ignores this column automatically.")
        if c["type"] == "text":
            share = sum(ml.is_number(_clean_number(v)) for v in present) / max(len(present), 1)
            odd = [v for v in set(present) if v.strip().lower() in ODD_MISSING]
            if share >= 0.8:
                add("warn", f"'{name}' holds numbers stored as text", "Symbols like $, % or thousands commas stop it being read as numbers. Remove them so it can be used as a number.")
            if odd:
                add("warn", f"'{name}' uses {', '.join(repr(o) for o in odd[:3])} for missing", "MiniLearn only recognises blank, NA, NaN, null, none or ? as missing. Replace these with a blank cell.")
            if c["unique"] > 10 and c["unique"] > 0.5 * n and share < 0.8:
                add("info", f"'{name}' is different on almost every row", "Free text such as names cannot be used as it is, so it is skipped. Group it into a few categories if it matters.")
            elif c["unique"] > 10:
                add("info", f"'{name}' has {c['unique']} categories", "Text columns with more than 10 categories are skipped. Merge rare categories into 'other' to use it.")
    return out


def readiness(advice):
    if any(a["level"] == "bad" for a in advice):
        return "Needs cleanup"
    return "Usable, with a few things to tidy" if any(a["level"] == "warn" for a in advice) else "Ready to go"


def about(header, rows, info):
    """Short plain-language description of what is in the file, plus sensible things to predict."""
    n = len(rows)
    num = [c for c in info if c["type"] == "numeric"]
    txt = [c for c in info if c["type"] == "text"]
    names = lambda cs, k=6: ", ".join(c["name"] for c in cs[:k]) + (f" and {len(cs) - k} more" if len(cs) > k else "")
    lines = [f"Each of the {n} rows is one record, described by {len(header)} columns: {len(num)} numeric ({names(num)}) and {len(txt)} text ({names(txt)})." if txt
             else f"Each of the {n} rows is one record, described by {len(header)} numeric columns ({names(num)})."]
    spread = []
    for i, c in enumerate(info):
        present = [r[i] for r in rows if not ml.is_missing(r[i])]
        if c["type"] == "numeric" and c["unique"] > 10 and not ml.looks_like_id(present):
            v = np.array([float(r[i]) for r in rows if ml.is_number(r[i]) and not ml.is_missing(r[i])])
            if len(v) and v.mean():
                spread.append((v.std() / abs(v.mean()), c["name"], v))
    for _, name, v in sorted(spread, reverse=True)[:2]:
        lines.append(f"'{name}' averages {v.mean():,.2f} and ranges from {v.min():,.2f} to {v.max():,.2f}.")
    gaps = sorted((c for c in info if c["missing"]), key=lambda c: -c["missing"])
    lines.append(f"{len(gaps)} column(s) have gaps; the emptiest is '{gaps[0]['name']}' ({100 * gaps[0]['missing'] / n:.0f}% blank)." if gaps else "There are no missing values.")
    suggested = []
    for i, c in enumerate(info):
        present = [r[i] for r in rows if not ml.is_missing(r[i])]
        if c["type"] == "empty" or c["unique"] < 2 or (c["type"] == "numeric" and present and ml.looks_like_id(present)):
            continue
        if c["unique"] == 2:
            suggested.append(dict(name=c["name"], why="two outcomes (yes/no style): classification"))
        elif c["type"] == "numeric" and c["unique"] > 10 and not c["missing"] > 0.3 * n:
            suggested.append(dict(name=c["name"], why="a number that varies: regression"))
        elif c["type"] == "text" and 2 < c["unique"] <= 10:
            suggested.append(dict(name=c["name"], why=f"{c['unique']} categories: classification"))
    last = info[-1]["name"]
    suggested.sort(key=lambda s: ("yes/no" not in s["why"], s["name"] != last))
    return dict(summary=lines, suggested=suggested[:4])