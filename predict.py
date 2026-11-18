"""'Try a prediction': build the input form and answer for one new row."""
import numpy as np

import minilearn as ml


def form_spec(header, rows, target, names):
    """One input field per raw column the models actually used."""
    used = []
    for n in names:
        raw = n.partition("=")[0] if any(x.startswith(n.partition("=")[0] + "=") for x in names) else n
        if raw not in used:
            used.append(raw)
    spec = []
    for raw in used:
        i = header.index(raw)
        vals = [r[i] for r in rows if not ml.is_missing(r[i])]
        if any(n.startswith(raw + "=") for n in names):
            spec.append(dict(name=raw, kind="choice", options=sorted(set(vals)), default=max(set(vals), key=vals.count)))
        else:
            spec.append(dict(name=raw, kind="number", default=float(np.median([float(v) for v in vals]))))
    return spec


def make_model(task, name):
    base, _, setting = name.partition(" [")
    for n, make in ml.MODELS[task]:
        if n == base:
            for label, candidate in ml.TUNING.get(base, []):
                if label == setting.rstrip("]"):
                    return candidate
            return make
    raise ValueError("Unknown model.")


def predict_all(X, y, task, names, spec, inputs, best_name, labels):
    by_name = {s["name"]: s for s in spec}
    row = []
    for n in names:
        col, eq, cat = n.partition("=")
        s = by_name.get(col if eq and col in by_name else n)
        v = str(inputs.get(s["name"], "")).strip()
        if s["kind"] == "choice":
            row.append(1.0 if (v or s["default"]) == cat else 0.0)
        else:
            row.append(float(v) if v else s["default"])
    Xs, xs = ml.scale(X, np.array([row]))
    best_base = best_name.partition(" [")[0]
    out = []
    for n, make in ml.MODELS[task]:
        if n.startswith("Baseline"):
            continue
        model = make_model(task, best_name) if n == best_base else make
        p = model().fit(Xs, y).predict(xs)[0]
        out.append(dict(model=n, best=(n == best_base),
                        answer=str(labels[int(p)]) if task == "classification" else f"{float(p):,.2f}"))
    return sorted(out, key=lambda a: not a["best"])