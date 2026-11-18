"""The web layer. It only moves data between the website and minilearn.py.
All the machine learning lives in minilearn.py."""
import csv
import io
import os
import tempfile

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

import minilearn as ml
import explore as ex
import health as hl
import json
import predict as pr

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB
MAX_ROWS = 2000  # larger files are sampled, so training stays quick


def read_upload(upload):
    """Read the upload tolerantly: any common encoding and delimiter is turned into a plain comma CSV.
    minilearn reads from a file path, so the cleaned text is saved to a temporary file first."""
    raw, notes = upload.read(), []
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
        notes.append(dict(level="info", title="The file was not UTF-8", detail="It was read as Windows-1252 (Excel's usual encoding). If accented letters look wrong, save it as 'CSV UTF-8'."))
    first = text.split("\n", 1)[0]
    delim = max(",;\t|", key=first.count) if any(d in first for d in ",;\t|") else ","
    if delim != ",":
        word = {";": "semicolons", "\t": "tabs", "|": "pipes"}[delim]
        notes.append(dict(level="info", title=f"Columns were separated by {word}", detail="MiniLearn converted the file to a normal comma-separated one automatically."))
    table = list(csv.reader(io.StringIO(text), delimiter=delim))
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", newline="", encoding="utf-8") as tmp:
        csv.writer(tmp).writerows(table)
        path = tmp.name
    try:
        header, rows = ml.load_csv(path)
    finally:
        os.remove(path)
    dropped = len([r for r in table[1:] if r]) - len(rows)
    if dropped > 0:
        notes.append(dict(level="warn", title=f"{dropped} rows were skipped", detail="They had a different number of cells from the header (often a stray comma or a broken line). Fix those lines in the file."))
    return header, rows, notes


@app.errorhandler(Exception)
def on_error(err):
    """Always answer with JSON, so the page can show a message instead of failing."""
    if isinstance(err, HTTPException):
        text = "That file is larger than 25 MB." if err.code == 413 else err.description
        return jsonify(error=text), err.code
    return jsonify(error=f"Unexpected server error: {err}"), 500


def drop_columns(header, rows, drop):
    keep = [i for i, h in enumerate(header) if h not in drop]
    return [header[i] for i in keep], [[r[i] for i in keep] for r in rows]


@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    upload = request.files.get("file")
    if upload is None:
        return jsonify(error="Choose a CSV file first."), 400
    try:
        header, rows, notes = read_upload(upload)
    except Exception:
        return jsonify(error="That file could not be read as a CSV."), 400
    if len(header) < 2 or len(rows) == 0:
        return jsonify(error="The file needs a header row, at least two columns and some data rows."), 400

    columns = ml.profile_columns(header, rows)

    # Step 1: no target chosen yet, so only describe the data
    target = request.form.get("target")
    if not target:
        advice = hl.check(header, rows, columns, notes)
        return jsonify(rows=len(rows), columns=columns, advice=advice, readiness=hl.readiness(advice),
                       about=hl.about(header, rows, columns))

    # Step 2: train and rank
    if target not in header:
        return jsonify(error="That column is not in the file."), 400
    tune = request.form.get("tune", "1") == "1"
    drop = [c for c in request.form.get("exclude", "").split("|") if c in header and c != target]
    header, rows = drop_columns(header, rows, drop)
    task = ml.detect_task(columns, target)

    sampled = len(rows) > MAX_ROWS
    if sampled:
        keep = np.random.default_rng(0).choice(len(rows), MAX_ROWS, replace=False)
        rows = [rows[i] for i in keep]

    try:
        X, y, labels, names = ml.prepare_data(header, rows, target, task)
        if len(names) == 0:
            return jsonify(error="No usable feature columns were found."), 400
        if len(y) < 30:
            return jsonify(error=f"Only {len(y)} rows have a target value. At least 30 are needed for a fair comparison."), 400
        if task == "classification" and len(set(y)) > 50:
            return jsonify(error=f"'{target}' has {len(set(y))} different values, so it behaves like an ID and there is nothing sensible to learn. Pick a column with fewer than 50 categories, or a numeric column."), 400
        if task == "classification" and len(set(y)) < 2:
            return jsonify(error="The target column has only one distinct value, so there is nothing to predict."), 400
        report = ml.rank_models(X, y, task, names, tune)
    except Exception as err:
        return jsonify(error=f"Could not train on this file: {err}"), 400

    advice = []
    lost = len(rows) - len(y)
    if lost > 0.2 * len(rows):
        advice.append(dict(level="warn", title=f"{lost} rows have no '{target}' value", detail="They cannot be used to learn. Fill in the target, or expect fewer rows than in the file."))
    if task == "classification":
        counts = np.bincount(y)
        rare = [str(labels[i]) for i, c in enumerate(counts) if c < 5]
        if rare:
            advice.append(dict(level="warn", title="Some classes have fewer than 5 rows", detail=f"{', '.join(rare[:4])}: the models cannot learn these reliably. Merge them into a bigger class or collect more rows."))
        if counts.max() / len(y) > 0.85:
            advice.append(dict(level="warn", title="One class dominates", detail="Over 85% of rows share one answer, so accuracy looks high even for a lazy model. Judge models by macro F1 and the baseline gap."))
        if len(labels) > 20:
            advice.append(dict(level="warn", title=f"{len(labels)} different classes", detail="That many classes is hard to predict from this many rows. Group similar classes together."))
    report.update(advice=advice, form=pr.form_spec(header, rows, target, names), columns=columns, classes=labels, features=names, sampled=sampled)
    return jsonify(report)


@app.route("/api/explore", methods=["POST"])
def explore_route():
    upload = request.files.get("file")
    if upload is None:
        return jsonify(error="Choose a CSV file first."), 400
    try:
        header, rows, _ = read_upload(upload)
        info = ml.profile_columns(header, rows)
        mode = request.form.get("mode")
        a, b = request.form.get("a"), request.form.get("b")
        if mode == "overview":
            return jsonify(ex.overview(header, rows, info))
        if mode == "compare":
            if a not in header or b not in header:
                return jsonify(error="Pick two columns from the file."), 400
            return jsonify(ex.compare(header, rows, info, a, b))
        if mode == "groups":
            k = max(2, min(6, int(request.form.get("k", 3))))
            return jsonify(ex.groups(header, rows, info, k))
        return jsonify(error="Unknown goal."), 400
    except ValueError as err:
        return jsonify(error=str(err)), 400
    except Exception as err:
        return jsonify(error=f"Could not study this file: {err}"), 400


@app.route("/api/predict", methods=["POST"])
def predict_route():
    upload = request.files.get("file")
    if upload is None:
        return jsonify(error="Choose a CSV file first."), 400
    header, rows, _ = read_upload(upload)
    target = request.form.get("target")
    if target not in header:
        return jsonify(error="That column is not in the file."), 400
    header, rows = drop_columns(header, rows, [c for c in request.form.get("exclude", "").split("|") if c in header and c != target])
    task = ml.detect_task(ml.profile_columns(header, rows), target)
    X, y, labels, names = ml.prepare_data(header, rows, target, task)
    spec = pr.form_spec(header, rows, target, names)
    try:
        answers = pr.predict_all(X, y, task, names, spec, json.loads(request.form.get("inputs", "{}")), request.form.get("model", ""), labels)
    except ValueError:
        return jsonify(error="Please type plain numbers in the number boxes."), 400
    return jsonify(answers=answers)


if __name__ == "__main__":
    app.run(debug=True)