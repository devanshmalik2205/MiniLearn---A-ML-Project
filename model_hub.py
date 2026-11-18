"""Model Hub & Model Persistence Engine.
Manages saving, versioning, loading, inspecting, comparing, downloading, deleting,
and running zero-retraining predictions on persisted syllabus models.

Storage Architecture:
- SQLite database (`saved_models/model_hub.db`): stores structured searchable metadata,
  versioning, dataset info, training configuration, preprocessing schema, and evaluation metrics.
- Safe JSON+NumPy model artifacts (`saved_models/<model_id>.artifact.json`): stores exact fitted
  model parameters (weights, support vectors/margins, k-NN reference matrices, Naive Bayes MLE parameters,
  or Decision Tree / Random Forest node structures) and exact fitted preprocessing parameters
  (imputation medians, categorical one-hot mappings, and scaler mean/std vectors).
  Uses HMAC-SHA256 integrity verification and never executes arbitrary code (no unsafe pickle).
"""
import datetime
import hashlib
import hmac
import json
import os
import re
import sqlite3
import sys
import time
import uuid

import numpy as np

import evaluation as ev
import minilearn as ml
import predict as pr

HUB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
DB_PATH = os.path.join(HUB_DIR, "model_hub.db")
HMAC_KEY = b"minilearn-academic-model-hub-v1-integrity-key"

ALGORITHM_DESCRIPTIONS = {
    "Baseline (most common class)": "Naive classification benchmark that always predicts the majority class from the training set.",
    "Baseline (average value)": "Naive regression benchmark that always predicts the mean target value from the training set.",
    "Linear regression": "Ordinary Least Squares (OLS) linear model that fits a hyperplane minimising sum of squared residuals.",
    "Ridge regression": "L2-regularised linear regression that penalises large weights (alpha * ||w||^2) to reduce variance and handle correlated features.",
    "Logistic regression": "Linear probabilistic classifier trained via gradient descent on cross-entropy loss using the logistic sigmoid function.",
    "Linear SVM": "Maximum-margin Support Vector Machine classifier trained via sub-gradient descent on hinge loss with L2 regularisation.",
    "KNN": "Instance-based K-Nearest Neighbours classifier that assigns the majority class among the k closest scaled training observations.",
    "KNN regressor": "Instance-based K-Nearest Neighbours regressor that predicts the average target value of the k closest scaled training observations.",
    "Naive Bayes": "Gaussian Bayesian classifier applying Bayes' rule under conditional feature independence with Maximum Likelihood mean and variance estimates.",
    "Decision tree": "Non-linear hierarchical classifier grown by greedily selecting feature-threshold splits that minimise Gini impurity.",
    "Decision tree regressor": "Non-linear hierarchical regressor grown by greedily selecting feature-threshold splits that minimise within-node variance.",
    "Random forest": "Bagged ensemble of randomised Decision Trees (bootstrapped rows + random feature subsets per split) that votes across trees.",
    "Random forest regressor": "Bagged ensemble of randomised regression trees whose predictions are averaged to reduce variance."
}


def _ensure_storage():
    os.makedirs(HUB_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                base_algorithm TEXT NOT NULL,
                full_model_name TEXT NOT NULL,
                version INTEGER NOT NULL,
                task_type TEXT NOT NULL,
                problem_subtype TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                target_column TEXT NOT NULL,
                n_rows_used INTEGER NOT NULL,
                n_dev_rows INTEGER NOT NULL,
                n_test_rows INTEGER NOT NULL,
                n_raw_features INTEGER NOT NULL,
                n_encoded_features INTEGER NOT NULL,
                hyperparameters_json TEXT NOT NULL,
                preprocessing_json TEXT NOT NULL,
                training_info_json TEXT NOT NULL,
                evaluation_metrics_json TEXT NOT NULL,
                primary_metric_name TEXT NOT NULL,
                primary_metric_value REAL NOT NULL,
                secondary_metric_name TEXT NOT NULL,
                secondary_metric_value REAL NOT NULL,
                cv_mean REAL NOT NULL,
                cv_std REAL NOT NULL,
                training_duration_s REAL NOT NULL,
                selection_status TEXT NOT NULL,
                artifact_filename TEXT NOT NULL,
                artifact_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def _safe_model_id(model_id):
    if not model_id or not re.match(r"^[a-f0-9\-]{16,64}$", str(model_id)):
        raise ValueError("Invalid model identifier format.")
    return str(model_id)


def _compute_hmac(payload_bytes):
    return hmac.new(HMAC_KEY, payload_bytes, hashlib.sha256).hexdigest()


def _serialize_model_object(model):
    """Convert a fitted MiniLearn model instance into a pure JSON-serializable dictionary."""
    cls_name = model.__class__.__name__
    if cls_name == "Baseline":
        return {
            "class": "Baseline",
            "task": model.task,
            "value": float(model.value) if model.task == "regression" else int(model.value),
            "n_classes": int(getattr(model, "n_classes", 0)),
            "priors": model.priors.tolist() if hasattr(model, "priors") else None
        }
    if cls_name == "RidgeRegression":
        return {
            "class": "RidgeRegression",
            "alpha": float(model.alpha),
            "w": model.w.tolist()
        }
    if cls_name == "LogisticRegression":
        return {
            "class": "LogisticRegression",
            "lr": float(model.lr),
            "epochs": int(model.epochs),
            "classes": model.classes.tolist(),
            "n_classes": int(model.n_classes),
            "W": model.W.tolist()
        }
    if cls_name == "LinearSVM":
        return {
            "class": "LinearSVM",
            "lr": float(model.lr),
            "epochs": int(model.epochs),
            "reg": float(model.reg),
            "classes": model.classes.tolist(),
            "n_classes": int(model.n_classes),
            "W": model.W.tolist()
        }
    if cls_name == "KNN":
        return {
            "class": "KNN",
            "task": model.task,
            "k": int(model.k),
            "n_classes": int(getattr(model, "n_classes", 0)),
            "X": model.X.tolist(),
            "y": model.y.tolist()
        }
    if cls_name == "GaussianNaiveBayes":
        return {
            "class": "GaussianNaiveBayes",
            "classes": model.classes.tolist(),
            "n_classes": int(model.n_classes),
            "means": model.means.tolist(),
            "vars": model.vars.tolist(),
            "priors": model.priors.tolist()
        }
    if cls_name == "DecisionTree":
        return {
            "class": "DecisionTree",
            "task": model.task,
            "max_depth": int(model.max_depth),
            "min_samples": int(model.min_samples),
            "max_features": int(model.max_features) if model.max_features is not None else None,
            "seed": int(model.seed),
            "n_classes": int(model.n_classes),
            "root": _clean_tree_node(model.root)
        }
    if cls_name == "RandomForest":
        return {
            "class": "RandomForest",
            "task": model.task,
            "n_trees": int(model.n_trees),
            "max_depth": int(model.max_depth),
            "n_classes": int(model.n_classes),
            "trees": [_serialize_model_object(t) for t in model.trees]
        }
    raise ValueError(f"Unsupported model class for serialization: {cls_name}")


def _clean_tree_node(node):
    if "value" in node:
        out = {
            "value": node["value"],
            "samples": int(node.get("samples", 0)),
            "impurity": float(node.get("impurity", 0.0))
        }
        if "probs" in node:
            out["probs"] = [float(x) for x in node["probs"]]
        return out
    return {
        "feature": int(node["feature"]),
        "threshold": float(node["threshold"]),
        "samples": int(node.get("samples", 0)),
        "impurity": float(node.get("impurity", 0.0)),
        "left": _clean_tree_node(node["left"]),
        "right": _clean_tree_node(node["right"])
    }


def _deserialize_model_object(state):
    """Reconstruct an exact fitted MiniLearn model instance without retraining."""
    if not isinstance(state, dict) or "class" not in state:
        raise ValueError("Corrupted model state: missing class descriptor.")
    cls_name = state["class"]

    if cls_name == "Baseline":
        m = ml.Baseline(state["task"])
        m.value = state["value"]
        m.n_classes = int(state.get("n_classes", 0))
        if state.get("priors") is not None:
            m.priors = np.array(state["priors"], dtype=float)
        return m

    if cls_name == "RidgeRegression":
        m = ml.RidgeRegression(alpha=float(state["alpha"]))
        m.w = np.array(state["w"], dtype=float)
        return m

    if cls_name == "LogisticRegression":
        m = ml.LogisticRegression(lr=float(state["lr"]), epochs=int(state["epochs"]))
        m.classes = np.array(state["classes"], dtype=int)
        m.n_classes = int(state["n_classes"])
        m.W = np.array(state["W"], dtype=float)
        return m

    if cls_name == "LinearSVM":
        m = ml.LinearSVM(lr=float(state["lr"]), epochs=int(state["epochs"]), reg=float(state["reg"]))
        m.classes = np.array(state["classes"], dtype=int)
        m.n_classes = int(state["n_classes"])
        m.W = np.array(state["W"], dtype=float)
        return m

    if cls_name == "KNN":
        m = ml.KNN(state["task"], k=int(state["k"]))
        m.n_classes = int(state.get("n_classes", 0))
        m.X = np.array(state["X"], dtype=float)
        m.y = np.array(state["y"], dtype=int if state["task"] == "classification" else float)
        return m

    if cls_name == "GaussianNaiveBayes":
        m = ml.GaussianNaiveBayes()
        m.classes = np.array(state["classes"], dtype=int)
        m.n_classes = int(state["n_classes"])
        m.means = np.array(state["means"], dtype=float)
        m.vars = np.array(state["vars"], dtype=float)
        m.priors = np.array(state["priors"], dtype=float)
        return m

    if cls_name == "DecisionTree":
        m = ml.DecisionTree(
            state["task"],
            max_depth=int(state["max_depth"]),
            min_samples=int(state["min_samples"]),
            max_features=state.get("max_features"),
            seed=int(state.get("seed", 0))
        )
        m.n_classes = int(state.get("n_classes", 0))
        if not isinstance(state.get("root"), dict):
            raise ValueError("Corrupted DecisionTree artifact: missing root tree structure.")
        m.root = state["root"]
        return m

    if cls_name == "RandomForest":
        m = ml.RandomForest(state["task"], n_trees=int(state["n_trees"]), max_depth=int(state["max_depth"]))
        m.n_classes = int(state.get("n_classes", 0))
        m.trees = [_deserialize_model_object(t) for t in state["trees"]]
        if len(m.trees) == 0:
            raise ValueError("Corrupted RandomForest artifact: empty tree list.")
        return m

    raise ValueError(f"Unknown model class in artifact: {cls_name}")


def _extract_hyperparameters(base_name, setting, model):
    params = {"setting_label": setting}
    cls_name = model.__class__.__name__
    if cls_name == "RidgeRegression":
        params["alpha"] = float(model.alpha)
        params["fit_intercept"] = True
    elif cls_name == "LogisticRegression":
        params["learning_rate"] = float(model.lr)
        params["epochs"] = int(model.epochs)
        params["optimizer"] = "Batch Gradient Descent"
    elif cls_name == "LinearSVM":
        params["regularization_reg"] = float(model.reg)
        params["learning_rate"] = float(model.lr)
        params["epochs"] = int(model.epochs)
    elif cls_name == "KNN":
        params["n_neighbors_k"] = int(model.k)
        params["distance_metric"] = "Standardized Euclidean (L2)"
    elif cls_name == "GaussianNaiveBayes":
        params["var_smoothing"] = 1e-9
        params["distribution"] = "Gaussian (MLE mean & variance)"
    elif cls_name == "DecisionTree":
        params["max_depth"] = int(model.max_depth)
        params["min_samples_leaf"] = int(model.min_samples)
        params["criterion"] = "Gini Impurity" if model.task == "classification" else "Squared Error (Variance)"
    elif cls_name == "RandomForest":
        params["n_trees"] = int(model.n_trees)
        params["max_depth"] = int(model.max_depth)
        params["bootstrap"] = True
        params["feature_subsampling"] = "sqrt(n_features)"
    return params


def _build_preprocessing_pipeline(header, rows, target, task, names, X_dev):
    """Capture the exact fitted preprocessing pipeline:
    1. Raw input feature schema (order, type, median for numeric, valid categories & mode for categorical).
    2. Encoded feature column order (`names`).
    3. Exact fitted standardization parameters (`mean` and `std` vectors computed on `X_dev`).
    """
    spec = pr.form_spec(header, rows, target, names)
    mean_vec = X_dev.mean(axis=0)
    std_vec = X_dev.std(axis=0)
    std_safe = np.where(std_vec == 0, 1.0, std_vec)

    raw_features = []
    for s in spec:
        col_idx = header.index(s["name"])
        vals = [r[col_idx] for r in rows if not ml.is_missing(r[col_idx])]
        if s["kind"] == "choice":
            raw_features.append({
                "name": s["name"],
                "kind": "categorical",
                "options": s["options"],
                "default_mode": s["default"],
                "missing_strategy": f"Impute with training mode ('{s['default']}') or zero-vector if unseen category"
            })
        else:
            nums = [float(v) for v in vals if ml.is_number(v)]
            raw_features.append({
                "name": s["name"],
                "kind": "numeric",
                "median": float(s["default"]),
                "min_observed": round(float(min(nums)), 4) if nums else 0.0,
                "max_observed": round(float(max(nums)), 4) if nums else 0.0,
                "default_median": float(s["default"]),
                "missing_strategy": f"Impute missing cells with training median ({s['default']:.4g})"
            })

    return {
        "raw_features": raw_features,
        "form_spec": spec,
        "encoded_feature_names": list(names),
        "scaler": {
            "type": "StandardScaler (Z-score standardization fitted on 80% development split)",
            "mean": [float(x) for x in mean_vec],
            "std": [float(x) for x in std_safe]
        },
        "target_encoding": {
            "target_column": target,
            "task": task
        }
    }


def get_next_version(base_algorithm, dataset_name, target_column):
    _ensure_storage()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT MAX(version) FROM models WHERE base_algorithm = ? AND dataset_name = ? AND target_column = ?",
            (base_algorithm, dataset_name, target_column)
        ).fetchone()
        return int(row[0] or 0) + 1


def save_model_to_hub(header, rows, dataset_name, target, force_task, tune, exclude_cols, selected_model_name, custom_name=None, overwrite_model_id=None):
    """Train/reproduce the exact model configuration from the evaluation pipeline and persist
    its fitted model artifact, fitted preprocessing pipeline, dataset metadata, and evaluation metrics."""
    _ensure_storage()

    # Compute dataset fingerprint
    raw_signature = f"{dataset_name}|{len(rows)}|{','.join(header)}|{target}|{'|'.join(sorted(exclude_cols))}"
    dataset_id = hashlib.sha256(raw_signature.encode("utf-8")).hexdigest()[:12]

    drop = [c for c in exclude_cols if c in header and c != target]
    keep = [i for i, h in enumerate(header) if h not in drop]
    sub_header = [header[i] for i in keep]
    sub_rows = [[r[i] for i in keep] for r in rows]

    columns = ml.profile_columns(sub_header, sub_rows)
    task = force_task if force_task in ("regression", "classification") else ml.detect_task(columns, target)

    sampled = len(sub_rows) > 2000
    if sampled:
        keep_rows = np.random.default_rng(0).choice(len(sub_rows), 2000, replace=False)
        sub_rows = [sub_rows[i] for i in keep_rows]

    X, y, labels, names = ml.prepare_data(sub_header, sub_rows, target, task)
    if len(names) == 0 or len(y) < 30:
        raise ValueError("Dataset does not have enough valid rows or features to train and save a model.")

    # Use the exact same 80/20 split seed (42) as rank_models
    rng = np.random.default_rng(42)
    locked = ml.make_folds(y, 5, rng, task == "classification")[0]
    dev = np.setdiff1d(np.arange(len(y)), locked)

    chosen = ml.choose_settings(ml.MODELS[task], X[dev], y[dev], task, tune)
    results = []
    for item in chosen:
        res = ml.cross_validate(item["make"], X[dev], y[dev], task)
        res.update(
            name=item["name"], base=item["base"], setting=item["setting"], make=item["make"],
            default_score=item.get("default_score"), tuned_score=item.get("tuned_score"),
            tuning_history=item.get("tuning_history", [])
        )
        results.append(res)

    is_imbalanced = (task == "classification" and float(np.bincount(y[dev]).max() / len(dev)) > 0.70)
    if is_imbalanced:
        results.sort(key=lambda r: (r["extra"] - 0.15 * r["extra_spread"], r["score"]), reverse=True)
    else:
        results.sort(key=lambda r: (r["score"] - 0.1 * r["spread"], r["extra"]), reverse=True)

    best_overall = results[0]

    # Match the requested model to save (by full name or base name)
    target_item = next((r for r in results if r["name"] == selected_model_name or r["base"] == selected_model_name), best_overall)

    # Fit preprocessing and model strictly on the 80% dev split and evaluate on the 20% test split
    X_train, X_test = ml.scale(X[dev], X[locked])
    t0 = time.time()
    fitted_model = target_item["make"]().fit(X_train, y[dev])
    fit_duration = time.time() - t0

    y_pred_test = fitted_model.predict(X_test)
    class_labels = [str(c) for c in labels] if (task == "classification" and labels is not None) else (
        [str(i) for i in range(int(y.max()) + 1)] if task == "classification" else []
    )

    if task == "classification":
        score_mat, score_kind = fitted_model.predict_scores(X_test) if hasattr(fitted_model, "predict_scores") else (None, "none")
        eval_detail = ev._classification_metrics_detail(y[locked], y_pred_test, score_mat, score_kind, class_labels)
        eval_metrics = {
            "task": "classification",
            "accuracy": eval_detail["accuracy"],
            "precision": eval_detail["macro_precision"],
            "recall": eval_detail["macro_recall"],
            "f1_score": eval_detail["macro_f1"],
            "weighted_f1": eval_detail["weighted_f1"],
            "roc_auc": eval_detail["roc_auc"],
            "avg_precision": eval_detail["avg_precision"],
            "confusion_matrix": eval_detail["confusion"],
            "confusion_narrative": eval_detail["confusion_narrative"],
            "per_class": eval_detail["per_class"],
            "class_labels": class_labels
        }
        primary_name = "Macro F1"
        primary_val = float(eval_detail["macro_f1"])
        sec_name = "Accuracy"
        sec_val = float(eval_detail["accuracy"])
        problem_subtype = "Binary Classification" if len(class_labels) == 2 else f"Multiclass Classification ({len(class_labels)} classes)"
    else:
        eval_detail = ev._regression_metrics_detail(y[locked], y_pred_test)
        mse_val = round(float(((y[locked] - y_pred_test) ** 2).mean()), 4)
        eval_metrics = {
            "task": "regression",
            "r2": eval_detail["r2"],
            "mae": eval_detail["mae"],
            "mse": mse_val,
            "rmse": eval_detail["rmse"],
            "mean_residual": eval_detail["mean_residual"],
            "median_residual": eval_detail["median_residual"],
            "std_residual": eval_detail["std_residual"],
            "narrative": eval_detail["narrative"]
        }
        primary_name = "R²"
        primary_val = float(eval_detail["r2"])
        sec_name = "RMSE"
        sec_val = float(eval_detail["rmse"])
        problem_subtype = "Supervised Regression"

    # Permutation feature importance for this saved model
    main_metric_fn = ml.METRICS[task][0]
    drops = ml.permutation_importance(fitted_model, X_test, y[locked], main_metric_fn, repeats=5)
    top_order = np.argsort(drops)[::-1]
    feature_importance = [[names[j], round(float(drops[j]), 4)] for j in top_order]
    eval_metrics["feature_importance"] = feature_importance

    # Versioning logic: never overwrite unless overwrite_model_id is explicitly provided
    base_alg = target_item["base"]
    if overwrite_model_id:
        model_id = _safe_model_id(overwrite_model_id)
        existing = get_model_details(model_id)
        version = existing["version"]
    else:
        model_id = str(uuid.uuid4())
        version = get_next_version(base_alg, dataset_name, target)

    display_name = (custom_name.strip() if custom_name and custom_name.strip() else f"{base_alg} – Version {version}")
    selection_status = "Selected Best Model" if target_item["name"] == best_overall["name"] else ("Statistically Tied with Best" if target_item.get("tie") else "Alternative Evaluated Candidate")

    preprocessing_pipeline = _build_preprocessing_pipeline(sub_header, sub_rows, target, task, names, X[dev])

    # Capture a compact held-out evaluation background sample (up to 120 test rows) so saved models
    # can compute Permutation Importance, Partial Dependence Plots (PDP), and sample-row Local Explanations
    # after application restart without requiring the CSV to be re-uploaded.
    t_idx = sub_header.index(target)
    valid_sub_rows = [r for r in sub_rows if not ml.is_missing(r[t_idx])]
    if task == "regression":
        valid_sub_rows = [r for r in valid_sub_rows if ml.is_number(r[t_idx])]
    eval_pick = locked[:min(120, len(locked))]
    raw_feat_names = [rf["name"] for rf in preprocessing_pipeline["raw_features"]]
    raw_feat_indices = [sub_header.index(fn) for fn in raw_feat_names]
    sample_test_records = []
    for row_i in eval_pick[:25]:
        r_obj = {sub_header[j]: valid_sub_rows[int(row_i)][j] for j in raw_feat_indices}
        actual_val = valid_sub_rows[int(row_i)][t_idx]
        sample_test_records.append({"row_index": int(row_i) + 1, "inputs": r_obj, "actual": actual_val})

    eval_reference = {
        "X_test_scaled": [[round(float(v), 5) for v in row] for row in X_test[:min(120, len(X_test))]],
        "y_test": [int(v) if task == "classification" else round(float(v), 5) for v in y[locked][:min(120, len(locked))]],
        "sample_records": sample_test_records
    }

    hyperparams = _extract_hyperparameters(base_alg, target_item["setting"], fitted_model)
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    training_info = {
        "model_name": display_name,
        "base_algorithm": base_alg,
        "full_model_name": target_item["name"],
        "algorithm_description": ALGORITHM_DESCRIPTIONS.get(base_alg, "Syllabus machine learning model."),
        "training_datetime": created_at,
        "cv_duration_seconds": round(float(target_item["seconds"]), 4),
        "final_fit_duration_seconds": round(float(fit_duration), 4),
        "train_test_split": f"80% Development ({len(dev)} rows) / 20% Locked Held-Out Test ({len(locked)} rows), random seed = 42",
        "cv_configuration": "3 Repeats × 5-Fold Stratified Cross-Validation (15 folds total on 80% dev split)" if task == "classification" else "3 Repeats × 5-Fold Cross-Validation (15 folds total on 80% dev split)",
        "tuning_enabled": bool(tune),
        "tuning_history": target_item.get("tuning_history", []),
        "random_seed": 42,
        "cv_score_name": ml.METRICS[task][1],
        "cv_mean": round(float(target_item["score"]), 4),
        "cv_std": round(float(target_item["spread"]), 4),
        "cv_min": round(float(target_item["min_score"]), 4),
        "cv_max": round(float(target_item["max_score"]), 4),
        "cv_extra_name": ml.METRICS[task][3],
        "cv_extra_mean": round(float(target_item["extra"]), 4),
        "cv_train_score": round(float(target_item["train_score"]), 4),
        "generalization_gap": round(float(target_item["train_score"] - target_item["score"]), 4),
        "selection_status": selection_status,
        "environment": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "serialization_format": "MiniLearn Safe JSON+NumPy Artifact v1 (HMAC-SHA256 verified)"
        }
    }

    dataset_info = {
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "total_rows_used": int(len(y)),
        "dev_rows": int(len(dev)),
        "test_rows": int(len(locked)),
        "sampled_to_2000": bool(sampled),
        "target_column": target,
        "task_type": task,
        "problem_subtype": problem_subtype,
        "raw_feature_names": [rf["name"] for rf in preprocessing_pipeline["raw_features"]],
        "encoded_feature_names": list(names),
        "excluded_columns": drop
    }

    # Build Artifact File
    serialized_model = _serialize_model_object(fitted_model)
    artifact_core = {
        "format": "minilearn_model_artifact_v1",
        "model_id": model_id,
        "display_name": display_name,
        "version": version,
        "created_at": created_at,
        "dataset_info": dataset_info,
        "training_info": training_info,
        "hyperparameters": hyperparams,
        "preprocessing_pipeline": preprocessing_pipeline,
        "evaluation_metrics": eval_metrics,
        "eval_reference": eval_reference,
        "class_labels": class_labels,
        "model_state": serialized_model
    }
    core_bytes = json.dumps(artifact_core, sort_keys=True).encode("utf-8")
    signature = _compute_hmac(core_bytes)
    artifact_envelope = {
        "integrity_hmac_sha256": signature,
        "artifact": artifact_core
    }
    artifact_filename = f"{model_id}.artifact.json"
    artifact_path = os.path.join(HUB_DIR, artifact_filename)
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact_envelope, f, indent=2)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO models (
                model_id, display_name, base_algorithm, full_model_name, version,
                task_type, problem_subtype, dataset_name, dataset_id, target_column,
                n_rows_used, n_dev_rows, n_test_rows, n_raw_features, n_encoded_features,
                hyperparameters_json, preprocessing_json, training_info_json, evaluation_metrics_json,
                primary_metric_name, primary_metric_value, secondary_metric_name, secondary_metric_value,
                cv_mean, cv_std, training_duration_s, selection_status,
                artifact_filename, artifact_sha256, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id, display_name, base_alg, target_item["name"], version,
            task, problem_subtype, dataset_name, dataset_id, target,
            int(len(y)), int(len(dev)), int(len(locked)),
            len(preprocessing_pipeline["raw_features"]), len(names),
            json.dumps(hyperparams), json.dumps(preprocessing_pipeline),
            json.dumps(training_info), json.dumps(eval_metrics),
            primary_name, primary_val, sec_name, sec_val,
            round(float(target_item["score"]), 4), round(float(target_item["spread"]), 4),
            round(float(target_item["seconds"]), 4), selection_status,
            artifact_filename, signature, created_at
        ))
        conn.commit()

    return get_model_details(model_id)


def list_saved_models():
    """Return all saved models plus dashboard summary statistics."""
    _ensure_storage()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM models ORDER BY created_at DESC, version DESC").fetchall()

    models = []
    for r in rows:
        artifact_path = os.path.join(HUB_DIR, r["artifact_filename"])
        artifact_ok = os.path.exists(artifact_path)
        models.append({
            "model_id": r["model_id"],
            "display_name": r["display_name"],
            "base_algorithm": r["base_algorithm"],
            "full_model_name": r["full_model_name"],
            "version": r["version"],
            "task_type": r["task_type"],
            "problem_subtype": r["problem_subtype"],
            "dataset_name": r["dataset_name"],
            "dataset_id": r["dataset_id"],
            "target_column": r["target_column"],
            "n_rows_used": r["n_rows_used"],
            "n_dev_rows": r["n_dev_rows"],
            "n_test_rows": r["n_test_rows"],
            "n_raw_features": r["n_raw_features"],
            "n_encoded_features": r["n_encoded_features"],
            "primary_metric_name": r["primary_metric_name"],
            "primary_metric_value": r["primary_metric_value"],
            "secondary_metric_name": r["secondary_metric_name"],
            "secondary_metric_value": r["secondary_metric_value"],
            "cv_mean": r["cv_mean"],
            "cv_std": r["cv_std"],
            "training_duration_s": r["training_duration_s"],
            "selection_status": r["selection_status"],
            "created_at": r["created_at"],
            "status": "Ready (Verified)" if artifact_ok else "Artifact Missing",
            "evaluation_metrics": json.loads(r["evaluation_metrics_json"]),
            "hyperparameters": json.loads(r["hyperparameters_json"])
        })

    cls_models = [m for m in models if m["task_type"] == "classification"]
    reg_models = [m for m in models if m["task_type"] == "regression"]
    best_cls = max(cls_models, key=lambda x: x["primary_metric_value"], default=None)
    best_reg = max(reg_models, key=lambda x: x["primary_metric_value"], default=None)

    summary = {
        "total_models": len(models),
        "classification_count": len(cls_models),
        "regression_count": len(reg_models),
        "most_recent": models[0] if models else None,
        "best_classification": best_cls,
        "best_regression": best_reg
    }
    return {"summary": summary, "models": models}


def get_model_details(model_id):
    """Retrieve complete stored metadata and verify artifact health for a single model."""
    _ensure_storage()
    mid = _safe_model_id(model_id)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        r = conn.execute("SELECT * FROM models WHERE model_id = ?", (mid,)).fetchone()
    if r is None:
        raise KeyError("Saved model was not found in the Model Hub.")

    artifact_path = os.path.join(HUB_DIR, r["artifact_filename"])
    artifact_status = "Verified & Ready"
    artifact_size_kb = 0.0
    if not os.path.exists(artifact_path):
        artifact_status = "Missing Artifact File"
    else:
        artifact_size_kb = round(os.path.getsize(artifact_path) / 1024.0, 2)
        try:
            load_model_artifact(mid)
        except Exception as err:
            artifact_status = f"Corrupted ({err})"

    return {
        "model_id": r["model_id"],
        "display_name": r["display_name"],
        "base_algorithm": r["base_algorithm"],
        "full_model_name": r["full_model_name"],
        "version": r["version"],
        "task_type": r["task_type"],
        "problem_subtype": r["problem_subtype"],
        "dataset_name": r["dataset_name"],
        "dataset_id": r["dataset_id"],
        "target_column": r["target_column"],
        "n_rows_used": r["n_rows_used"],
        "n_dev_rows": r["n_dev_rows"],
        "n_test_rows": r["n_test_rows"],
        "n_raw_features": r["n_raw_features"],
        "n_encoded_features": r["n_encoded_features"],
        "hyperparameters": json.loads(r["hyperparameters_json"]),
        "preprocessing": json.loads(r["preprocessing_json"]),
        "training_info": json.loads(r["training_info_json"]),
        "evaluation_metrics": json.loads(r["evaluation_metrics_json"]),
        "primary_metric_name": r["primary_metric_name"],
        "primary_metric_value": r["primary_metric_value"],
        "secondary_metric_name": r["secondary_metric_name"],
        "secondary_metric_value": r["secondary_metric_value"],
        "cv_mean": r["cv_mean"],
        "cv_std": r["cv_std"],
        "training_duration_s": r["training_duration_s"],
        "selection_status": r["selection_status"],
        "created_at": r["created_at"],
        "artifact_info": {
            "artifact_id": r["model_id"],
            "status": artifact_status,
            "size_kb": artifact_size_kb,
            "sha256_hmac": r["artifact_sha256"],
            "format": "MiniLearn Safe JSON+NumPy Artifact v1"
        }
    }


def load_model_artifact(model_id):
    """Load, validate HMAC integrity, and reconstruct the exact trained model and preprocessing
    pipeline from disk WITHOUT retraining."""
    _ensure_storage()
    mid = _safe_model_id(model_id)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        r = conn.execute("SELECT artifact_filename, artifact_sha256 FROM models WHERE model_id = ?", (mid,)).fetchone()
    if r is None:
        raise KeyError("Saved model was not found in the Model Hub.")

    artifact_path = os.path.join(HUB_DIR, r["artifact_filename"])
    if not os.path.exists(artifact_path):
        raise ValueError("Model artifact file is missing from storage.")

    try:
        with open(artifact_path, "r", encoding="utf-8") as f:
            envelope = json.load(f)
    except Exception as err:
        raise ValueError(f"Model artifact file is corrupted (invalid JSON): {err}")

    if not isinstance(envelope, dict) or "artifact" not in envelope or "integrity_hmac_sha256" not in envelope:
        raise ValueError("Model artifact file is corrupted: missing envelope structure.")

    core = envelope["artifact"]
    core_bytes = json.dumps(core, sort_keys=True).encode("utf-8")
    expected_sig = _compute_hmac(core_bytes)
    if not hmac.compare_digest(expected_sig, str(envelope["integrity_hmac_sha256"])):
        raise ValueError("Model artifact integrity check failed (HMAC-SHA256 mismatch; file was modified or corrupted).")

    preproc = core.get("preprocessing_pipeline", {})
    scaler = preproc.get("scaler", {})
    names = preproc.get("encoded_feature_names", [])
    if not names or len(scaler.get("mean", [])) != len(names) or len(scaler.get("std", [])) != len(names):
        raise ValueError("Corrupted preprocessing pipeline: scaler dimensions do not match encoded feature schema.")

    model_obj = _deserialize_model_object(core.get("model_state"))
    return {
        "model": model_obj,
        "artifact": core,
        "preprocessing": preproc,
        "class_labels": core.get("class_labels", [])
    }


def predict_with_saved_model(model_id, raw_inputs, strict=False):
    """Run prediction using the exact saved model and its saved preprocessing pipeline.
    Never retrains or refits any component.
    Validates feature schema, data types, missing values, and unseen categories."""
    loaded = load_model_artifact(model_id)
    model = loaded["model"]
    core = loaded["artifact"]
    preproc = loaded["preprocessing"]
    class_labels = loaded["class_labels"]
    task = core["dataset_info"]["task_type"]

    if not isinstance(raw_inputs, dict):
        raise ValueError("Prediction inputs must be provided as a JSON key-value object.")

    raw_features = preproc["raw_features"]
    expected_names = [rf["name"] for rf in raw_features]
    expected_set = set(expected_names)
    provided_set = set(raw_inputs.keys())

    extra_features = sorted(provided_set - expected_set)
    missing_features = [fn for fn in expected_names if fn not in raw_inputs or ml.is_missing(str(raw_inputs.get(fn, "")))]

    if strict and (extra_features or missing_features):
        msgs = []
        if missing_features:
            msgs.append(f"Missing required feature(s): {', '.join(missing_features)}.")
        if extra_features:
            msgs.append(f"Unexpected extra feature(s) not in model schema: {', '.join(extra_features)}.")
        raise ValueError("Schema mismatch: " + " ".join(msgs))

    warnings = []
    if extra_features:
        warnings.append(f"Ignored {len(extra_features)} extra input field(s) not used by this model: {', '.join(extra_features)}.")

    cleaned_raw = {}
    for rf in raw_features:
        fname = rf["name"]
        raw_val = str(raw_inputs.get(fname, "")).strip()
        if rf["kind"] == "numeric":
            if ml.is_missing(raw_val):
                cleaned_raw[fname] = float(rf["default_median"])
                warnings.append(f"'{fname}' was blank; imputed using saved training median ({rf['default_median']:.4g}).")
            else:
                if not ml.is_number(raw_val):
                    raise ValueError(f"Invalid numeric input for '{fname}': '{raw_val}' is not a valid number.")
                num_v = float(raw_val)
                if not np.isfinite(num_v):
                    raise ValueError(f"Invalid numeric input for '{fname}': value must be finite.")
                cleaned_raw[fname] = num_v
                if num_v < rf["min_observed"] or num_v > rf["max_observed"]:
                    warnings.append(f"Note: '{fname}' = {num_v:g} lies outside the training range [{rf['min_observed']:g}, {rf['max_observed']:g}].")
        else:
            valid_opts = rf["options"]
            if ml.is_missing(raw_val):
                cleaned_raw[fname] = rf["default_mode"]
                warnings.append(f"'{fname}' was blank; imputed using saved training mode ('{rf['default_mode']}').")
            elif raw_val not in valid_opts:
                if strict:
                    raise ValueError(f"Invalid category '{raw_val}' for '{fname}'. Allowed categories: {', '.join(valid_opts)}.")
                cleaned_raw[fname] = "__UNSEEN__"
                warnings.append(f"Unseen category '{raw_val}' for '{fname}' (expected one of: {', '.join(valid_opts)}). Encoded as all-zeros for one-hot columns.")
            else:
                cleaned_raw[fname] = raw_val

    # Preserve exact encoded feature order and apply exact saved StandardScaler parameters
    encoded_names = preproc["encoded_feature_names"]
    rf_by_name = {rf["name"]: rf for rf in raw_features}
    row_vec = []
    for enc_name in encoded_names:
        col, eq, cat = enc_name.partition("=")
        if eq and col in rf_by_name and rf_by_name[col]["kind"] == "categorical":
            row_vec.append(1.0 if cleaned_raw[col] == cat else 0.0)
        else:
            row_vec.append(float(cleaned_raw[enc_name]))

    x_arr = np.array([row_vec], dtype=float)
    mean_arr = np.array(preproc["scaler"]["mean"], dtype=float)
    std_arr = np.array(preproc["scaler"]["std"], dtype=float)
    x_scaled = (x_arr - mean_arr) / np.where(std_arr == 0, 1.0, std_arr)

    pred_raw = model.predict(x_scaled)[0]
    result = {
        "model_id": core["model_id"],
        "display_name": core["display_name"],
        "base_algorithm": core["training_info"]["base_algorithm"],
        "version": core["version"],
        "task_type": task,
        "target_column": core["dataset_info"]["target_column"],
        "retrained": False,
        "warnings": warnings,
        "applied_inputs": cleaned_raw
    }

    if task == "classification":
        pred_idx = int(pred_raw)
        pred_label = str(class_labels[pred_idx]) if 0 <= pred_idx < len(class_labels) else str(pred_idx)
        result["prediction"] = pred_label
        if hasattr(model, "predict_scores"):
            scores, score_kind = model.predict_scores(x_scaled)
            if score_kind == "probability":
                probs = scores[0]
                result["probabilities"] = [
                    {"class": str(class_labels[i]), "probability": round(float(probs[i]), 4)}
                    for i in range(min(len(class_labels), len(probs)))
                ]
                result["probability_supported"] = True
            else:
                result["probabilities"] = None
                result["probability_supported"] = False
                result["probability_note"] = f"{core['training_info']['base_algorithm']} uses continuous margin decision scores rather than calibrated class probabilities."
        else:
            result["probabilities"] = None
            result["probability_supported"] = False
    else:
        val = float(pred_raw)
        result["prediction"] = f"{val:,.2f}"
        result["prediction_numeric"] = round(val, 4)

    return result


def compare_saved_models(model_ids):
    """Compare multiple saved models, checking task compatibility and dataset/split comparability."""
    if not model_ids or len(model_ids) < 2:
        raise ValueError("Select at least two saved models to compare.")
    details = [get_model_details(mid) for mid in model_ids]
    tasks = {d["task_type"] for d in details}
    if len(tasks) > 1:
        raise ValueError("Cannot compare Classification and Regression models in the same ranking because their evaluation metrics (Accuracy/F1 vs R²/RMSE) are incompatible.")

    task = details[0]["task_type"]
    datasets = {(d["dataset_id"], d["target_column"]) for d in details}
    same_dataset_and_target = (len(datasets) == 1)

    comparability_warning = None
    if not same_dataset_and_target:
        comparability_warning = (
            "Caution — Cross-Dataset Comparison: The selected models were trained on different datasets, feature sets, or target columns. "
            "Metrics calculated on different datasets are shown for reference only and do not represent a head-to-head competition on the same test observations."
        )

    rows = []
    for d in details:
        em = d["evaluation_metrics"]
        rows.append({
            "model_id": d["model_id"],
            "display_name": d["display_name"],
            "base_algorithm": d["base_algorithm"],
            "version": d["version"],
            "dataset_name": d["dataset_name"],
            "dataset_id": d["dataset_id"],
            "target_column": d["target_column"],
            "task_type": d["task_type"],
            "n_rows_used": d["n_rows_used"],
            "n_raw_features": d["n_raw_features"],
            "cv_mean": d["cv_mean"],
            "cv_std": d["cv_std"],
            "training_duration_s": d["training_duration_s"],
            "created_at": d["created_at"],
            "metrics": em
        })

    # Sort by primary metric
    if task == "classification":
        rows.sort(key=lambda r: (r["metrics"].get("f1_score", 0), r["metrics"].get("accuracy", 0)), reverse=True)
    else:
        rows.sort(key=lambda r: r["metrics"].get("r2", -1e9), reverse=True)

    return {
        "task_type": task,
        "same_dataset_and_target": same_dataset_and_target,
        "comparability_warning": comparability_warning,
        "models": rows
    }


def delete_saved_model(model_id):
    """Delete a single specific model version and its artifact file without affecting other versions."""
    _ensure_storage()
    mid = _safe_model_id(model_id)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        r = conn.execute("SELECT display_name, version, artifact_filename FROM models WHERE model_id = ?", (mid,)).fetchone()
        if r is None:
            raise KeyError("Saved model was not found.")
        conn.execute("DELETE FROM models WHERE model_id = ?", (mid,))
        conn.commit()

    artifact_path = os.path.join(HUB_DIR, r["artifact_filename"])
    if os.path.exists(artifact_path):
        try:
            os.remove(artifact_path)
        except OSError:
            pass
    return {"deleted_model_id": mid, "display_name": r["display_name"], "version": r["version"]}


def get_download_payload(model_id, kind="artifact"):
    """Return a clean download payload ('artifact' or 'metadata') with no internal server file paths."""
    details = get_model_details(model_id)
    safe_slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", details["display_name"]).strip("_").lower()
    if kind == "metadata":
        payload = {
            "download_type": "MiniLearn Model Metadata & Evaluation Summary",
            "included_contents": "Dataset summary, feature schema, hyperparameters, training configuration, cross-validation scores, and held-out test evaluation metrics.",
            "model_overview": {
                "model_id": details["model_id"],
                "display_name": details["display_name"],
                "base_algorithm": details["base_algorithm"],
                "version": details["version"],
                "task_type": details["task_type"],
                "problem_subtype": details["problem_subtype"],
                "created_at": details["created_at"],
                "selection_status": details["selection_status"]
            },
            "dataset_info": {
                "dataset_name": details["dataset_name"],
                "dataset_id": details["dataset_id"],
                "target_column": details["target_column"],
                "n_rows_used": details["n_rows_used"],
                "n_dev_rows": details["n_dev_rows"],
                "n_test_rows": details["n_test_rows"],
                "n_raw_features": details["n_raw_features"],
                "n_encoded_features": details["n_encoded_features"]
            },
            "hyperparameters": details["hyperparameters"],
            "preprocessing_pipeline": details["preprocessing"],
            "training_info": details["training_info"],
            "evaluation_metrics": details["evaluation_metrics"],
            "artifact_info": details["artifact_info"]
        }
        return payload, f"{safe_slug}_metadata.json"
    else:
        mid = _safe_model_id(model_id)
        artifact_path = os.path.join(HUB_DIR, f"{mid}.artifact.json")
        if not os.path.exists(artifact_path):
            raise ValueError("Saved model artifact file is missing.")
        with open(artifact_path, "r", encoding="utf-8") as f:
            envelope = json.load(f)
        envelope["download_explanation"] = (
            "Complete MiniLearn Saved Model Artifact. Includes the HMAC-SHA256 integrity signature, "
            "fitted model parameters (weights/trees/centroids), fitted Z-score StandardScaler vectors, "
            "median/mode imputation rules, feature schema, and evaluation metadata."
        )
        return envelope, f"{safe_slug}_v{details['version']}.artifact.json"
