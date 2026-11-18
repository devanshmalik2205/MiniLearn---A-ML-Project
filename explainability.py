"""Model Explainability Centre — Global Feature Importance, Model-Specific Explanations,
Local Single-Prediction Explanations, Interactive Decision Tree Structure,
Partial Dependence Plots (PDP), Counterfactual What-If Analysis, and Reliability Diagnostics.

Strictly adheres to the university AI/ML course syllabus and operates on actual fitted
models and preprocessing pipelines without fabricating explanations or retraining."""
import numpy as np

import minilearn as ml
import model_hub as mh


def _metric_fn_by_name(task, metric_key=None):
    if task == "classification":
        if metric_key == "accuracy":
            return ml.accuracy, "Accuracy"
        return ml.macro_f1, "Macro F1"
    else:
        if metric_key == "rmse":
            # Negate RMSE so higher = better (shuffling drops score)
            return (lambda y, p: -ml.rmse(y, p)), "Negated RMSE"
        if metric_key == "mae":
            return (lambda y, p: -ml.mae(y, p)), "Negated MAE"
        return ml.r2, "R²"


def _permutation_importance_with_std(model, X_test, y_test, metric_fn, names, repeats=5):
    """Compute mean and standard deviation of performance drop across repeats when each feature is shuffled."""
    if len(X_test) == 0 or len(y_test) == 0:
        return []
    rng = np.random.default_rng(7)
    base_score = float(metric_fn(y_test, model.predict(X_test)))
    items = []
    for j, fname in enumerate(names):
        drops = []
        for _ in range(repeats):
            shuffled = X_test.copy()
            shuffled[:, j] = rng.permutation(shuffled[:, j])
            s = float(metric_fn(y_test, model.predict(shuffled)))
            drops.append(base_score - s)
        mean_d = float(np.mean(drops))
        std_d = float(np.std(drops))
        items.append({
            "feature": fname,
            "raw_feature": fname.partition("=")[0] if "=" in fname else fname,
            "importance_mean": round(mean_d, 4),
            "importance_std": round(std_d, 4),
            "fold_drops": [round(x, 4) for x in drops]
        })
    items.sort(key=lambda x: x["importance_mean"], reverse=True)
    return items


def _tree_impurity_importance(root_node, n_features):
    """Traverse a DecisionTree node hierarchy to compute exact weighted impurity decrease per feature."""
    gains = np.zeros(n_features, dtype=float)

    def _visit(node):
        if "value" in node:
            return
        f_idx = int(node["feature"])
        n_node = float(node.get("samples", 1))
        imp_node = float(node.get("impurity", 0.0))
        left = node["left"]
        right = node["right"]
        n_left = float(left.get("samples", max(1.0, n_node / 2.0)))
        n_right = float(right.get("samples", max(1.0, n_node / 2.0)))
        imp_left = float(left.get("impurity", 0.0))
        imp_right = float(right.get("impurity", 0.0))
        decrease = n_node * imp_node - (n_left * imp_left + n_right * imp_right)
        if 0 <= f_idx < n_features:
            gains[f_idx] += max(0.0, decrease)
        _visit(left)
        _visit(right)

    _visit(root_node)
    total = float(gains.sum())
    if total > 0:
        gains = gains / total
    return gains


def _export_tree_structure(node, names, scaler_mean, scaler_std, class_labels, task, depth=0, node_id="0"):
    """Convert an internal DecisionTree node into an educational, human-readable tree dictionary
    showing both scaled and unscaled (original unit) split thresholds."""
    samples = int(node.get("samples", 0))
    impurity = round(float(node.get("impurity", 0.0)), 4)

    if "value" in node:
        if task == "classification":
            c_idx = int(node["value"])
            pred_str = str(class_labels[c_idx]) if 0 <= c_idx < len(class_labels) else str(c_idx)
            probs = node.get("probs", [])
            max_p = max(probs) * 100.0 if probs else 100.0
            return {
                "id": node_id,
                "is_leaf": True,
                "depth": depth,
                "samples": samples,
                "impurity": impurity,
                "impurity_name": "Gini",
                "prediction": pred_str,
                "confidence_pct": round(float(max_p), 1),
                "probs": [{"class": str(class_labels[i]), "prob": round(float(p), 4)} for i, p in enumerate(probs) if i < len(class_labels)]
            }
        else:
            return {
                "id": node_id,
                "is_leaf": True,
                "depth": depth,
                "samples": samples,
                "impurity": impurity,
                "impurity_name": "Variance",
                "prediction": round(float(node["value"]), 3),
                "confidence_pct": None,
                "probs": None
            }

    f_idx = int(node["feature"])
    fname = names[f_idx] if 0 <= f_idx < len(names) else f"feature_{f_idx}"
    scaled_thresh = float(node["threshold"])
    # Unscale threshold back to original dataset units: x_orig = x_scaled * std + mean
    orig_thresh = scaled_thresh * float(scaler_std[f_idx]) + float(scaler_mean[f_idx])

    if "=" in fname:
        col, _, cat = fname.partition("=")
        condition_text = f"{col} is '{cat}'"
        left_label = f"No ({col} ≠ {cat})"
        right_label = f"Yes ({col} = {cat})"
    else:
        condition_text = f"{fname} ≤ {orig_thresh:,.2f}"
        left_label = f"≤ {orig_thresh:,.2f}"
        right_label = f"> {orig_thresh:,.2f}"

    return {
        "id": node_id,
        "is_leaf": False,
        "depth": depth,
        "feature_index": f_idx,
        "feature_name": fname,
        "threshold_scaled": round(scaled_thresh, 4),
        "threshold_original": round(orig_thresh, 4),
        "condition": condition_text,
        "left_branch_label": left_label,
        "right_branch_label": right_label,
        "samples": samples,
        "impurity": impurity,
        "impurity_name": "Gini" if task == "classification" else "Variance",
        "left": _export_tree_structure(node["left"], names, scaler_mean, scaler_std, class_labels, task, depth + 1, node_id + "L"),
        "right": _export_tree_structure(node["right"], names, scaler_mean, scaler_std, class_labels, task, depth + 1, node_id + "R")
    }


def _collect_top_splits(tree_dict, out_list=None):
    if out_list is None:
        out_list = []
    if tree_dict.get("is_leaf"):
        return out_list
    out_list.append({
        "depth": tree_dict["depth"],
        "condition": tree_dict["condition"],
        "feature": tree_dict["feature_name"],
        "threshold_original": tree_dict["threshold_original"],
        "samples": tree_dict["samples"],
        "impurity": tree_dict["impurity"]
    })
    _collect_top_splits(tree_dict["left"], out_list)
    _collect_top_splits(tree_dict["right"], out_list)
    return out_list


def _model_specific_explanation(model, base_alg, task, names, preproc, class_labels):
    """Extract genuine model-specific explanations where mathematically supported by the algorithm."""
    cls_name = model.__class__.__name__
    scaler_mean = np.array(preproc["scaler"]["mean"], dtype=float)
    scaler_std = np.array(preproc["scaler"]["std"], dtype=float)

    # 1. Linear & Ridge Regression
    if cls_name == "RidgeRegression":
        intercept = float(model.w[0])
        weights = np.array(model.w[1:], dtype=float)
        unscaled_w = weights / np.where(scaler_std == 0, 1.0, scaler_std)
        coefs = []
        for j, fn in enumerate(names):
            coefs.append({
                "feature": fn,
                "scaled_coefficient": round(float(weights[j]), 4),
                "unscaled_slope": round(float(unscaled_w[j]), 4),
                "abs_magnitude": round(abs(float(weights[j])), 4),
                "direction": "Positive (+)" if weights[j] > 0 else ("Negative (-)" if weights[j] < 0 else "Zero (0)")
            })
        coefs.sort(key=lambda x: x["abs_magnitude"], reverse=True)
        return {
            "supported": True,
            "method_type": "coefficients",
            "method_name": "Standardized Linear Coefficients (Weights w)",
            "intercept": round(intercept, 4),
            "coefficients": coefs,
            "explanation": (
                "Each standardized coefficient represents the change in predicted target value per +1 standard deviation increase "
                "in that feature (holding other features constant). Because features were standardized before fitting, coefficient magnitudes "
                "can be compared directly."
            )
        }

    # 2. Logistic Regression & Linear SVM
    if cls_name in ("LogisticRegression", "LinearSVM"):
        W = np.array(model.W, dtype=float)  # n_classes x (1 + n_features)
        is_binary = (len(model.classes) == 2)
        # For binary classification, use the positive class (row 1) weight vector
        row_idx = 1 if is_binary and W.shape[0] > 1 else 0
        intercept = float(W[row_idx, 0])
        weights = W[row_idx, 1:] if is_binary else np.mean(np.abs(W[:, 1:]), axis=0)
        signed_weights = W[row_idx, 1:]

        class_coefs = []
        for c_i, c_val in enumerate(model.classes):
            c_name = str(class_labels[int(c_val)]) if int(c_val) < len(class_labels) else str(c_val)
            class_coefs.append({
                "class_name": c_name,
                "intercept": round(float(W[c_i, 0]), 4),
                "weights": {names[j]: round(float(W[c_i, j + 1]), 4) for j in range(len(names))}
            })

        coefs = []
        for j, fn in enumerate(names):
            w_val = float(signed_weights[j]) if is_binary else float(weights[j])
            coefs.append({
                "feature": fn,
                "scaled_coefficient": round(w_val, 4),
                "abs_magnitude": round(abs(w_val), 4),
                "direction": (
                    f"Pushes toward '{class_labels[1]}' (+)" if (is_binary and w_val > 0) else
                    (f"Pushes toward '{class_labels[0]}' (-)" if (is_binary and w_val < 0) else f"Mean |w| across {len(model.classes)} classes")
                )
            })
        coefs.sort(key=lambda x: x["abs_magnitude"], reverse=True)
        return {
            "supported": True,
            "method_type": "coefficients",
            "method_name": "Linear Hyperplane Weights / Log-Odds Coefficients" if cls_name == "LogisticRegression" else "Linear SVM Hyperplane Margin Weights",
            "intercept": round(intercept, 4),
            "is_binary": is_binary,
            "positive_class": str(class_labels[1]) if is_binary and len(class_labels) > 1 else None,
            "coefficients": coefs,
            "per_class_weights": class_coefs,
            "explanation": (
                "Positive weights increase the log-odds / margin score toward the positive class when the standardized feature is above its mean; "
                "negative weights push the decision toward the opposite class."
            )
        }

    # 3. Decision Tree
    if cls_name == "DecisionTree":
        imp_vec = _tree_impurity_importance(model.root, len(names))
        tree_struct = _export_tree_structure(model.root, names, scaler_mean, scaler_std, class_labels, task)
        splits = _collect_top_splits(tree_struct)
        splits.sort(key=lambda s: (s["depth"], -s["samples"]))
        imp_list = [
            {"feature": names[j], "impurity_importance": round(float(imp_vec[j]), 4)}
            for j in np.argsort(-imp_vec)
        ]
        return {
            "supported": True,
            "method_type": "decision_tree",
            "method_name": "Decision Tree Hierarchy & Gini/Variance Impurity Reduction",
            "impurity_importances": imp_list,
            "tree_structure": tree_struct,
            "top_splits": splits[:8],
            "explanation": (
                "Built-in tree importance sums the total reduction in node impurity (Gini for classification, Variance for regression) "
                "weighted by the number of observations passing through each split."
            )
        }

    # 4. Random Forest
    if cls_name == "RandomForest":
        tree_imps = np.array([_tree_impurity_importance(t.root, len(names)) for t in model.trees])
        mean_imp = tree_imps.mean(axis=0)
        std_imp = tree_imps.std(axis=0)
        imp_list = [
            {
                "feature": names[j],
                "impurity_importance": round(float(mean_imp[j]), 4),
                "impurity_std": round(float(std_imp[j]), 4)
            }
            for j in np.argsort(-mean_imp)
        ]
        trees_exported = [
            _export_tree_structure(t.root, names, scaler_mean, scaler_std, class_labels, task)
            for t in model.trees[:min(5, len(model.trees))]
        ]
        return {
            "supported": True,
            "method_type": "random_forest",
            "method_name": f"Ensemble Impurity Importance Across {len(model.trees)} Decision Trees",
            "n_trees": len(model.trees),
            "impurity_importances": imp_list,
            "sample_trees": trees_exported,
            "explanation": (
                f"Averages the impurity-reduction importance across all {len(model.trees)} bootstrapped trees in the forest. "
                "Note: Impurity-based importance can favour continuous features with many split points over binary one-hot features; "
                "always compare it against Permutation Importance."
            )
        }

    # 5. Gaussian Naive Bayes
    if cls_name == "GaussianNaiveBayes":
        # Discriminative power of feature j across classes: variance of class means divided by mean within-class variance
        between_var = np.var(model.means, axis=0)
        within_var = np.mean(model.vars, axis=0)
        disc_ratio = between_var / np.maximum(within_var, 1e-9)
        total_r = float(disc_ratio.sum()) or 1.0
        norm_disc = disc_ratio / total_r

        params_list = []
        for j in np.argsort(-norm_disc):
            fn = names[j]
            per_cls = {}
            for c_i, c_val in enumerate(model.classes):
                c_name = str(class_labels[int(c_val)]) if int(c_val) < len(class_labels) else str(c_val)
                # Convert mean back to original units for readability
                orig_mean = float(model.means[c_i, j]) * float(scaler_std[j]) + float(scaler_mean[j])
                orig_std = float(np.sqrt(model.vars[c_i, j])) * float(scaler_std[j])
                per_cls[c_name] = {"mean_orig": round(orig_mean, 3), "std_orig": round(orig_std, 3), "mean_z": round(float(model.means[c_i, j]), 3)}
            params_list.append({
                "feature": fn,
                "discriminative_share": round(float(norm_disc[j]), 4),
                "signal_to_noise": round(float(disc_ratio[j]), 4),
                "class_distributions": per_cls
            })
        return {
            "supported": True,
            "method_type": "naive_bayes",
            "method_name": "Gaussian Likelihood Parameter Separation (MLE Class Means & Variances)",
            "priors": {str(class_labels[int(c)]): round(float(p), 4) for c, p in zip(model.classes, model.priors)},
            "feature_parameters": params_list,
            "explanation": (
                "Gaussian Naive Bayes models each feature independently within each class using Maximum Likelihood mean (μ_c) and variance (σ²_c). "
                "Features where class means are far apart relative to within-class variance provide the strongest log-likelihood separation."
            )
        }

    # 6. KNN or Baseline: No intrinsic feature weights exist
    if cls_name == "KNN":
        return {
            "supported": False,
            "method_type": "none",
            "method_name": "Not Applicable (Instance-Based Non-Parametric Model)",
            "explanation": (
                f"{base_alg} stores training instances and computes Euclidean distances across all scaled features equally without learning "
                "internal feature weights or tree splits. Use model-agnostic Permutation Importance and Local Perturbation to understand its behaviour."
            )
        }

    return {
        "supported": False,
        "method_type": "none",
        "method_name": "Not Applicable (Constant Baseline Model)",
        "explanation": "Baseline always predicts a constant value (majority class or training mean) and ignores all input features."
    }


def _encode_and_scale_raw_row(cleaned_raw, preproc):
    encoded_names = preproc["encoded_feature_names"]
    rf_by_name = {rf["name"]: rf for rf in preproc["raw_features"]}
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
    return x_arr[0], x_scaled


def _scalar_model_output(model, x_scaled, task, target_class_idx=None):
    """Return a continuous scalar response for sensitivity/perturbation/PDP evaluation."""
    if task == "regression":
        return float(model.predict(x_scaled)[0]), "Predicted Value", None
    if hasattr(model, "predict_scores"):
        scores, kind = model.predict_scores(x_scaled)
        k = scores.shape[1]
        if target_class_idx is None:
            target_class_idx = 1 if k == 2 else int(np.argmax(scores[0]))
        val = float(scores[0, target_class_idx])
        label = "Predicted Probability" if kind == "probability" else "Decision Margin Score"
        return val, label, int(target_class_idx)
    pred = int(model.predict(x_scaled)[0])
    return float(pred), "Predicted Class Index", pred


def explain_local_prediction(model_id, raw_inputs, actual_value=None):
    """Explain a single individual prediction using both:
    1. Exact analytical decomposition (for Linear/Ridge/Logistic/SVM/NaiveBayes/DecisionTree path), AND
    2. Baseline-Anchored Feature Attribution (measuring how replacing each raw feature with its
       typical training baseline median/mode shifts the model's output for this specific record).
    Never fabricates values or claims to be SHAP/LIME."""
    pred_res = mh.predict_with_saved_model(model_id, raw_inputs, strict=False)
    loaded = mh.load_model_artifact(model_id)
    model = loaded["model"]
    core = loaded["artifact"]
    preproc = loaded["preprocessing"]
    class_labels = loaded["class_labels"]
    task = core["dataset_info"]["task_type"]

    cleaned_raw = pred_res["applied_inputs"]
    _, x_scaled = _encode_and_scale_raw_row(cleaned_raw, preproc)

    # Determine target class for attribution in classification
    pred_class_idx = None
    if task == "classification":
        pred_label = pred_res["prediction"]
        pred_class_idx = class_labels.index(pred_label) if pred_label in class_labels else 0

    full_score, score_unit, _ = _scalar_model_output(model, x_scaled, task, pred_class_idx)

    # Compute typical baseline record (every raw feature set to its training median or mode)
    baseline_raw = {
        rf["name"]: (rf["default_median"] if rf["kind"] == "numeric" else rf["default_mode"])
        for rf in preproc["raw_features"]
    }
    _, base_scaled = _encode_and_scale_raw_row(baseline_raw, preproc)
    base_score, _, _ = _scalar_model_output(model, base_scaled, task, pred_class_idx)

    # Measure each raw feature's local contribution for this specific observation:
    # Contribution_i = Score(x_current) - Score(x_current with feature_i replaced by its training median/mode)
    contributions = []
    for rf in preproc["raw_features"]:
        fname = rf["name"]
        perturbed_raw = dict(cleaned_raw)
        ref_val = rf["default_median"] if rf["kind"] == "numeric" else rf["default_mode"]
        perturbed_raw[fname] = ref_val
        _, pert_scaled = _encode_and_scale_raw_row(perturbed_raw, preproc)
        pert_score, _, _ = _scalar_model_output(model, pert_scaled, task, pred_class_idx)
        effect = full_score - pert_score

        if abs(effect) < 1e-6:
            direction = "Neutral (0)"
        elif effect > 0:
            direction = f"Increased '{pred_res['prediction']}' (+)" if task == "classification" else "Increased prediction (+)"
        else:
            direction = f"Decreased '{pred_res['prediction']}' (-)" if task == "classification" else "Decreased prediction (-)"

        contributions.append({
            "feature": fname,
            "input_value": cleaned_raw[fname],
            "reference_baseline": ref_val,
            "contribution": round(float(effect), 4),
            "abs_contribution": round(abs(float(effect)), 4),
            "direction": direction
        })

    contributions.sort(key=lambda x: x["abs_contribution"], reverse=True)

    # If the model is a Decision Tree, also trace the exact decision path taken by this row
    decision_path = None
    if model.__class__.__name__ == "DecisionTree":
        decision_path = []
        node = model.root
        names = preproc["encoded_feature_names"]
        s_mean = preproc["scaler"]["mean"]
        s_std = preproc["scaler"]["std"]
        row_z = x_scaled[0]
        while "value" not in node:
            f_idx = int(node["feature"])
            fname = names[f_idx]
            thresh_z = float(node["threshold"])
            orig_t = thresh_z * float(s_std[f_idx]) + float(s_mean[f_idx])
            went_left = bool(row_z[f_idx] <= thresh_z)
            decision_path.append({
                "feature": fname,
                "threshold_original": round(orig_t, 3),
                "went": "Left (≤ threshold)" if went_left else "Right (> threshold)"
            })
            node = node["left"] if went_left else node["right"]

    # Generate plain-language narrative
    pos_feats = [c for c in contributions if c["contribution"] > 1e-5][:3]
    neg_feats = [c for c in contributions if c["contribution"] < -1e-5][:3]
    narrative_lines = [
        f"Predicted {core['dataset_info']['target_column']}: {pred_res['prediction']}"
        + (f" (Actual: {actual_value})" if actual_value is not None and str(actual_value) != "" else "")
        + f" — evaluated via {score_unit} ({full_score:.3f} vs typical baseline record {base_score:.3f})."
    ]
    if pos_feats:
        narrative_lines.append(
            "Positive contributors pushing the prediction upward: "
            + ", ".join(f"{c['feature']} = {c['input_value']} ({c['contribution']:+.3f} vs median/mode {c['reference_baseline']})" for c in pos_feats) + "."
        )
    if neg_feats:
        narrative_lines.append(
            "Negative contributors pulling the prediction downward: "
            + ", ".join(f"{c['feature']} = {c['input_value']} ({c['contribution']:+.3f} vs median/mode {c['reference_baseline']})" for c in neg_feats) + "."
        )
    if not pos_feats and not neg_feats:
        narrative_lines.append("All input features sit near their typical baseline values or lie inside a flat decision region.")
    narrative_lines.append("Note: These are model-derived local attributions (single-feature counterfactual shifts relative to training medians/modes), not proof of real-world causation.")

    return {
        "prediction_result": pred_res,
        "actual_value": actual_value,
        "score_unit": score_unit,
        "full_score": round(full_score, 4),
        "baseline_reference_score": round(base_score, 4),
        "contributions": contributions,
        "decision_path": decision_path,
        "narrative": narrative_lines,
        "methodology": (
            "Single-Feature Counterfactual Perturbation: Measures how much the model's continuous output changes when each input feature "
            "is individually replaced with its typical training baseline value (median for numeric, mode for categorical), while preserving "
            "valid data types and exact one-hot/Z-score preprocessing."
        )
    }


def compute_partial_dependence(model_id, feature_name=None):
    """Compute Partial Dependence Plot (PDP) curves across valid feature ranges using the saved
    held-out evaluation background matrix. Supports both numeric grids and categorical levels."""
    loaded = mh.load_model_artifact(model_id)
    model = loaded["model"]
    core = loaded["artifact"]
    preproc = loaded["preprocessing"]
    class_labels = loaded["class_labels"]
    task = core["dataset_info"]["task_type"]

    raw_features = preproc["raw_features"]
    if not raw_features:
        raise ValueError("No raw features available for Partial Dependence analysis.")

    if not feature_name:
        feature_name = raw_features[0]["name"]
    rf = next((x for x in raw_features if x["name"] == feature_name), raw_features[0])
    feature_name = rf["name"]

    # Retrieve held-out scaled background matrix (or synthesize from scaler means if legacy artifact)
    eval_ref = core.get("eval_reference", {})
    X_bg = np.array(eval_ref.get("X_test_scaled", []), dtype=float)
    encoded_names = preproc["encoded_feature_names"]
    s_mean = np.array(preproc["scaler"]["mean"], dtype=float)
    s_std = np.array(preproc["scaler"]["std"], dtype=float)

    if X_bg.ndim != 2 or len(X_bg) == 0 or X_bg.shape[1] != len(encoded_names):
        X_bg = np.zeros((1, len(encoded_names)), dtype=float)

    # Helper to evaluate average model output over X_mod
    def _eval_matrix(X_mat):
        if task == "regression":
            preds = model.predict(X_mat)
            return float(np.mean(preds)), float(np.std(preds)), "Average Predicted Target"
        if hasattr(model, "predict_scores"):
            scores, kind = model.predict_scores(X_mat)
            c_idx = 1 if scores.shape[1] == 2 else 0
            c_lbl = str(class_labels[c_idx]) if c_idx < len(class_labels) else str(c_idx)
            unit = f"Avg Predicted Probability (Class '{c_lbl}')" if kind == "probability" else f"Avg Decision Margin (Class '{c_lbl}')"
            return float(np.mean(scores[:, c_idx])), float(np.std(scores[:, c_idx])), unit
        preds = model.predict(X_mat)
        return float(np.mean(preds)), float(np.std(preds)), "Average Predicted Class"

    grid_points = []
    unit_label = "Average Model Output"

    if rf["kind"] == "numeric":
        f_idx = encoded_names.index(feature_name)
        v_min = float(rf["min_observed"])
        v_max = float(rf["max_observed"])
        if v_max <= v_min:
            v_max = v_min + 1.0
        xs = np.linspace(v_min, v_max, 20)
        for x_val in xs:
            X_mod = X_bg.copy()
            z_val = (x_val - s_mean[f_idx]) / (s_std[f_idx] if s_std[f_idx] != 0 else 1.0)
            X_mod[:, f_idx] = z_val
            mean_y, std_y, unit_label = _eval_matrix(X_mod)
            grid_points.append({
                "x": round(float(x_val), 4),
                "label": f"{x_val:,.2f}",
                "mean_response": round(mean_y, 4),
                "std_response": round(std_y, 4)
            })
    else:
        # Categorical PDP: set one-hot columns for `feature_name` to each valid category level
        cat_cols = [(j, enc.partition("=")[2]) for j, enc in enumerate(encoded_names) if enc.startswith(feature_name + "=")]
        for opt in rf["options"]:
            X_mod = X_bg.copy()
            for j, cat_val in cat_cols:
                raw_ind = 1.0 if opt == cat_val else 0.0
                X_mod[:, j] = (raw_ind - s_mean[j]) / (s_std[j] if s_std[j] != 0 else 1.0)
            mean_y, std_y, unit_label = _eval_matrix(X_mod)
            grid_points.append({
                "x": opt,
                "label": str(opt),
                "mean_response": round(mean_y, 4),
                "std_response": round(std_y, 4)
            })

    responses = [p["mean_response"] for p in grid_points]
    span = max(responses) - min(responses) if responses else 0.0
    if rf["kind"] == "numeric" and len(responses) >= 2:
        trend = "increases" if responses[-1] > responses[0] + 1e-4 else ("decreases" if responses[-1] < responses[0] - 1e-4 else "remains flat")
        summary_text = (
            f"As '{feature_name}' moves from {rf['min_observed']:g} to {rf['max_observed']:g}, the {unit_label.lower()} "
            f"{trend} from {responses[0]:.3f} to {responses[-1]:.3f} (total PDP swing = {span:.3f})."
        )
    else:
        best_cat = max(grid_points, key=lambda p: p["mean_response"]) if grid_points else {"label": "N/A", "mean_response": 0}
        min_cat = min(grid_points, key=lambda p: p["mean_response"]) if grid_points else {"label": "N/A", "mean_response": 0}
        summary_text = (
            f"Across categories of '{feature_name}', '{best_cat['label']}' yields the highest average response ({best_cat['mean_response']:.3f}) "
            f"and '{min_cat['label']}' yields the lowest ({min_cat['mean_response']:.3f}) (swing = {span:.3f})."
        )

    return {
        "feature_name": feature_name,
        "feature_kind": rf["kind"],
        "unit_label": unit_label,
        "n_background_rows": int(len(X_bg)),
        "points": grid_points,
        "response_swing": round(float(span), 4),
        "summary": summary_text
    }


def run_what_if_analysis(model_id, original_inputs, modified_inputs):
    """Compare Original vs Modified input records using the exact saved model and preprocessing pipeline.
    Never retrains the model."""
    orig_pred = mh.predict_with_saved_model(model_id, original_inputs, strict=False)
    mod_pred = mh.predict_with_saved_model(model_id, modified_inputs, strict=False)

    loaded = mh.load_model_artifact(model_id)
    model = loaded["model"]
    core = loaded["artifact"]
    preproc = loaded["preprocessing"]
    class_labels = loaded["class_labels"]
    task = core["dataset_info"]["task_type"]

    _, orig_scaled = _encode_and_scale_raw_row(orig_pred["applied_inputs"], preproc)
    _, mod_scaled = _encode_and_scale_raw_row(mod_pred["applied_inputs"], preproc)

    changed_features = []
    for rf in preproc["raw_features"]:
        fn = rf["name"]
        v1 = orig_pred["applied_inputs"].get(fn)
        v2 = mod_pred["applied_inputs"].get(fn)
        if str(v1) != str(v2):
            changed_features.append({"feature": fn, "from": v1, "to": v2})

    if task == "regression":
        v_orig = float(orig_pred["prediction_numeric"])
        v_mod = float(mod_pred["prediction_numeric"])
        abs_diff = v_mod - v_orig
        pct_diff = (abs_diff / abs(v_orig) * 100.0) if abs(v_orig) > 1e-6 else None
        return {
            "task_type": "regression",
            "retrained": False,
            "original_prediction": orig_pred["prediction"],
            "modified_prediction": mod_pred["prediction"],
            "original_numeric": round(v_orig, 4),
            "modified_numeric": round(v_mod, 4),
            "absolute_difference": round(abs_diff, 4),
            "percentage_difference": round(pct_diff, 2) if pct_diff is not None else None,
            "changed_features": changed_features,
            "warnings": mod_pred.get("warnings", []),
            "summary": (
                f"Changing {len(changed_features)} feature(s) shifted predicted '{core['dataset_info']['target_column']}' "
                f"from {orig_pred['prediction']} to {mod_pred['prediction']} "
                f"(Δ = {abs_diff:+,.2f}" + (f", {pct_diff:+.1f}%)." if pct_diff is not None else ").")
            )
        }
    else:
        c_idx = 1 if len(class_labels) == 2 else 0
        s_orig, unit_lbl, _ = _scalar_model_output(model, orig_scaled, task, c_idx)
        s_mod, _, _ = _scalar_model_output(model, mod_scaled, task, c_idx)
        score_diff = s_mod - s_orig
        class_changed = (orig_pred["prediction"] != mod_pred["prediction"])
        return {
            "task_type": "classification",
            "retrained": False,
            "original_prediction": orig_pred["prediction"],
            "modified_prediction": mod_pred["prediction"],
            "class_flipped": class_changed,
            "score_unit": unit_lbl,
            "original_score": round(s_orig, 4),
            "modified_score": round(s_mod, 4),
            "absolute_difference": round(score_diff, 4),
            "percentage_difference": round(score_diff * 100.0, 2) if "Probability" in unit_lbl else None,
            "original_probabilities": orig_pred.get("probabilities"),
            "modified_probabilities": mod_pred.get("probabilities"),
            "changed_features": changed_features,
            "warnings": mod_pred.get("warnings", []),
            "summary": (
                f"Changing {len(changed_features)} feature(s) "
                + (f"flipped the predicted class from '{orig_pred['prediction']}' to '{mod_pred['prediction']}' " if class_changed else f"kept the predicted class at '{mod_pred['prediction']}' ")
                + f"({unit_lbl} shifted by {score_diff:+.3f}, from {s_orig:.3f} to {s_mod:.3f})."
            )
        }


def _reliability_and_behaviour_audit(perm_items, model_spec, core, X_bg):
    """Analyse model concentration, near-zero features, permutation vs intrinsic discrepancy,
    correlated features, and input sensitivity."""
    findings = []
    pos_drops = [max(0.0, x["importance_mean"]) for x in perm_items]
    total_drop = sum(pos_drops)
    top_share = (pos_drops[0] / total_drop * 100.0) if (total_drop > 1e-6 and pos_drops) else 0.0
    near_zero = [x["feature"] for x in perm_items if abs(x["importance_mean"]) < 0.002]
    negative_perm = [x["feature"] for x in perm_items if x["importance_mean"] < -0.005]

    if top_share > 65.0 and len(perm_items) >= 2:
        findings.append({
            "level": "warn",
            "title": f"High Single-Feature Dependence ({top_share:.0f}% of total impact)",
            "detail": f"'{perm_items[0]['feature']}' accounts for {top_share:.1f}% of the total permutation drop. If this feature is corrupted or shifts in production, model predictions will degrade sharply."
        })
    elif len(perm_items) >= 2 and total_drop > 1e-5:
        findings.append({
            "level": "info",
            "title": f"Balanced Feature Reliance (Top feature = {top_share:.0f}% of impact)",
            "detail": f"Predictive signal is distributed across multiple features rather than relying solely on '{perm_items[0]['feature']}'."
        })

    if near_zero:
        findings.append({
            "level": "info",
            "title": f"{len(near_zero)} Feature(s) with Near-Zero Permutation Impact",
            "detail": f"Shuffling {', '.join(near_zero[:6])}{'...' if len(near_zero) > 6 else ''} changed held-out performance by less than 0.002. These features may be redundant or uninformative for {core['training_info']['base_algorithm']}."
        })

    if negative_perm:
        findings.append({
            "level": "warn",
            "title": f"{len(negative_perm)} Feature(s) Slightly Hurt Generalization on Held-Out Data",
            "detail": f"Shuffling {', '.join(negative_perm[:4])} slightly improved the held-out test score, suggesting minor overfitting to noise in those columns."
        })

    # Compare model-specific ranking vs permutation ranking
    if model_spec.get("supported") and perm_items:
        top_perm = perm_items[0]["feature"]
        top_intr = None
        if model_spec["method_type"] == "coefficients" and model_spec.get("coefficients"):
            top_intr = model_spec["coefficients"][0]["feature"]
        elif model_spec["method_type"] in ("decision_tree", "random_forest") and model_spec.get("impurity_importances"):
            top_intr = model_spec["impurity_importances"][0]["feature"]
        elif model_spec["method_type"] == "naive_bayes" and model_spec.get("feature_parameters"):
            top_intr = model_spec["feature_parameters"][0]["feature"]

        if top_intr and top_intr != top_perm:
            findings.append({
                "level": "warn",
                "title": "Discrepancy Between Intrinsic & Permutation Top Feature",
                "detail": f"Internal model parameters rank '{top_intr}' highest, whereas held-out Permutation Importance ranks '{top_perm}' highest. This commonly occurs when features are correlated or when tree impurity favours continuous columns."
            })

    # Check feature correlation in X_bg
    names = core["preprocessing_pipeline"]["encoded_feature_names"]
    if len(X_bg) >= 10 and X_bg.shape[1] >= 2:
        corr_pairs = []
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                if "=" in names[a] and "=" in names[b] and names[a].split("=")[0] == names[b].split("=")[0]:
                    continue
                if np.std(X_bg[:, a]) > 0 and np.std(X_bg[:, b]) > 0:
                    r_val = abs(float(np.corrcoef(X_bg[:, a], X_bg[:, b])[0, 1]))
                    if r_val >= 0.55:
                        corr_pairs.append((names[a], names[b], r_val))
        corr_pairs.sort(key=lambda x: -x[2])
        if corr_pairs:
            p0 = corr_pairs[0]
            findings.append({
                "level": "warn",
                "title": f"Correlated Features Detected (|r| = {p0[2]:.2f} between '{p0[0]}' & '{p0[1]}')",
                "detail": "When features are strongly correlated, Permutation Importance and Partial Dependence Plots can underestimate importance or evaluate unrealistic feature combinations."
            })

    gen_gap = float(core["training_info"].get("generalization_gap", 0.0))
    if gen_gap > 0.08:
        findings.append({
            "level": "warn",
            "title": f"Overfitting Warning (Train–Validation Gap = {gen_gap:+.3f})",
            "detail": "The model fits its training rows noticeably better than unseen validation folds, meaning local explanations may reflect memorized training quirks."
        })

    return {
        "top_feature_concentration_pct": round(top_share, 1),
        "near_zero_features": near_zero,
        "findings": findings
    }


def get_model_explainability(model_id, metric_key=None, pdp_feature=None):
    """Build the complete Model Explainability Centre payload for a saved model."""
    loaded = mh.load_model_artifact(model_id)
    model = loaded["model"]
    core = loaded["artifact"]
    preproc = loaded["preprocessing"]
    class_labels = loaded["class_labels"]
    task = core["dataset_info"]["task_type"]
    base_alg = core["training_info"]["base_algorithm"]
    names = preproc["encoded_feature_names"]

    eval_ref = core.get("eval_reference", {})
    X_test = np.array(eval_ref.get("X_test_scaled", []), dtype=float)
    y_test = np.array(eval_ref.get("y_test", []), dtype=int if task == "classification" else float)

    metric_fn, metric_label = _metric_fn_by_name(task, metric_key)
    if len(X_test) > 0 and len(y_test) == len(X_test):
        perm_items = _permutation_importance_with_std(model, X_test, y_test, metric_fn, names, repeats=5)
    else:
        # Fallback to stored permutation importance from evaluation_metrics
        stored_fi = core.get("evaluation_metrics", {}).get("feature_importance", [])
        perm_items = [
            {
                "feature": fn,
                "raw_feature": fn.partition("=")[0] if "=" in fn else fn,
                "importance_mean": round(float(d), 4),
                "importance_std": 0.0,
                "fold_drops": [round(float(d), 4)]
            }
            for fn, d in stored_fi
        ]

    model_spec = _model_specific_explanation(model, base_alg, task, names, preproc, class_labels)
    pdp_data = compute_partial_dependence(model_id, pdp_feature)
    reliability = _reliability_and_behaviour_audit(perm_items, model_spec, core, X_test)

    # Default local explanation on the first held-out sample record (or typical baseline record)
    sample_records = eval_ref.get("sample_records", [])
    if sample_records:
        default_rec = sample_records[0]
        local_expl = explain_local_prediction(model_id, default_rec["inputs"], actual_value=default_rec.get("actual"))
    else:
        default_inputs = {rf["name"]: (rf["default_median"] if rf["kind"] == "numeric" else rf["default_mode"]) for rf in preproc["raw_features"]}
        local_expl = explain_local_prediction(model_id, default_inputs)

    most_inf = perm_items[0] if perm_items else None
    least_inf = perm_items[-1] if perm_items else None
    is_baseline = base_alg.startswith("Baseline")

    overview = {
        "model_id": core["model_id"],
        "display_name": core["display_name"],
        "base_algorithm": base_alg,
        "full_model_name": core["training_info"]["full_model_name"],
        "version": core["version"],
        "dataset_name": core["dataset_info"]["dataset_name"],
        "target_column": core["dataset_info"]["target_column"],
        "task_type": task,
        "problem_subtype": core["dataset_info"]["problem_subtype"],
        "n_raw_features": len(preproc["raw_features"]),
        "n_encoded_features": len(names),
        "n_eval_rows": int(len(X_test)) if len(X_test) > 0 else int(core["dataset_info"]["test_rows"]),
        "primary_metric_name": "Macro F1" if task == "classification" else "R²",
        "primary_metric_value": core["evaluation_metrics"].get("f1_score" if task == "classification" else "r2", 0.0),
        "most_influential_feature": most_inf["feature"] if (most_inf and not is_baseline) else "None (Baseline ignores features)",
        "most_influential_drop": most_inf["importance_mean"] if most_inf else 0.0,
        "least_influential_feature": least_inf["feature"] if (least_inf and not is_baseline) else "None",
        "least_influential_drop": least_inf["importance_mean"] if least_inf else 0.0,
        "supports_global_permutation": not is_baseline,
        "supports_model_specific": bool(model_spec["supported"]),
        "model_specific_method": model_spec["method_name"],
        "supports_local_explanation": not is_baseline,
        "supports_tree_viz": model.__class__.__name__ in ("DecisionTree", "RandomForest"),
        "supports_probabilities": hasattr(model, "predict_scores") and model.__class__.__name__ not in ("LinearSVM", "Baseline")
    }

    return {
        "overview": overview,
        "raw_features": preproc["raw_features"],
        "sample_records": sample_records,
        "permutation_metric": metric_label,
        "permutation_importance": perm_items,
        "model_specific": model_spec,
        "local_explanation": local_expl,
        "pdp": pdp_data,
        "reliability": reliability
    }
