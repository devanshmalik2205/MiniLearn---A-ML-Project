"""Model Evaluation Centre — Comprehensive Evaluation, Diagnostics, Curves, Bootstrap CIs,
Learning Curves, Permutation Feature Importance Comparison, and Educational Explanations.
Strictly limited to the university AI/ML course syllabus."""
import numpy as np
import minilearn as ml


def _roc_curve_binary(y_true_bin, scores):
    """Compute exact False Positive Rate, True Positive Rate, and trapezoidal ROC-AUC
    from continuous model probabilities or decision scores."""
    pos = (y_true_bin == 1)
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None, None, None
    if np.all(scores == scores[0]):
        # Constant score (e.g. Baseline prior): diagonal line from (0,0) to (1,1), AUC = 0.5
        return [0.0, 1.0], [0.0, 1.0], 0.5

    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = pos[order].astype(float)

    distinct_indices = np.where(np.diff(s_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, len(y_sorted) - 1]

    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    tps = np.r_[0.0, tps]
    fps = np.r_[0.0, fps]

    fpr = fps / n_neg
    tpr = tps / n_pos
    auc = float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))

    # Downsample to at most 80 points for clean SVG rendering while preserving endpoints
    if len(fpr) > 80:
        idx = np.unique(np.linspace(0, len(fpr) - 1, 80).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return [round(float(x), 4) for x in fpr], [round(float(y), 4) for y in tpr], round(auc, 4)


def _pr_curve_binary(y_true_bin, scores):
    """Compute exact Recall, Precision, and Average Precision (AP) from continuous scores.
    Never confuses Average Precision with ROC-AUC."""
    pos = (y_true_bin == 1)
    n_pos = int(pos.sum())
    if n_pos == 0 or len(y_true_bin) == 0:
        return None, None, None
    base_rate = float(n_pos / len(y_true_bin))
    if np.all(scores == scores[0]):
        return [0.0, 1.0], [base_rate, base_rate], round(base_rate, 4)

    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = pos[order].astype(float)

    distinct_indices = np.where(np.diff(s_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, len(y_sorted) - 1]

    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / n_pos

    rec_full = np.r_[0.0, recall]
    prec_full = np.r_[1.0, precision]

    # Average precision: area under precision-recall step function: sum((R_n - R_{n-1}) * P_n)
    ap = float(np.sum((rec_full[1:] - rec_full[:-1]) * prec_full[1:]))

    if len(rec_full) > 80:
        idx = np.unique(np.linspace(0, len(rec_full) - 1, 80).astype(int))
        rec_full, prec_full = rec_full[idx], prec_full[idx]
    return [round(float(x), 4) for x in rec_full], [round(float(y), 4) for y in prec_full], round(ap, 4)


def _classification_metrics_detail(y_true, y_pred, score_mat, score_kind, class_labels):
    """Compute full class-wise precision, recall, F1, support, macro/weighted averages,
    confusion matrix (raw and row-normalized), ROC-AUC, and Precision-Recall curves."""
    n = len(y_true)
    k = len(class_labels)
    cm = np.zeros((k, k), dtype=int)
    for a, p in zip(y_true, y_pred):
        if 0 <= int(a) < k and 0 <= int(p) < k:
            cm[int(a), int(p)] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, (cm / np.maximum(row_sums, 1)) * 100.0, 0.0)

    per_class = []
    prec_list, rec_list, f1_list, sup_list = [], [], [], []
    for c in range(k):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        tn = int(n - tp - fp - fn)
        sup = int(tp + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class.append({
            "class_index": c,
            "class_name": str(class_labels[c]),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "support": sup,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        })
        if sup > 0:
            prec_list.append(prec)
            rec_list.append(rec)
            f1_list.append(f1)
            sup_list.append(sup)

    total_sup = max(sum(sup_list), 1)
    macro_p = float(np.mean(prec_list)) if prec_list else 0.0
    macro_r = float(np.mean(rec_list)) if rec_list else 0.0
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    weighted_p = float(sum(p * s for p, s in zip(prec_list, sup_list)) / total_sup)
    weighted_r = float(sum(r * s for r, s in zip(rec_list, sup_list)) / total_sup)
    weighted_f1 = float(sum(f * s for f, s in zip(f1_list, sup_list)) / total_sup)
    acc = float((y_true == y_pred).mean()) if n > 0 else 0.0

    # Binary positive class convention: class 1 (for k==2), else macro average over classes
    roc_data = {"available": False, "score_kind": score_kind, "auc": None, "curves": []}
    pr_data = {"available": False, "score_kind": score_kind, "ap": None, "curves": []}

    if score_mat is not None and score_mat.shape == (n, k):
        if k == 2:
            y_bin = (y_true == 1).astype(int)
            fpr, tpr, auc = _roc_curve_binary(y_bin, score_mat[:, 1])
            rec_c, prec_c, ap = _pr_curve_binary(y_bin, score_mat[:, 1])
            if fpr is not None:
                roc_data = {
                    "available": True,
                    "score_kind": score_kind,
                    "auc": auc,
                    "curves": [{"class_name": str(class_labels[1]), "fpr": fpr, "tpr": tpr, "auc": auc}]
                }
            if rec_c is not None:
                pr_data = {
                    "available": True,
                    "score_kind": score_kind,
                    "ap": ap,
                    "baseline_rate": round(float(y_bin.mean()), 4),
                    "curves": [{"class_name": str(class_labels[1]), "recall": rec_c, "precision": prec_c, "ap": ap}]
                }
        else:
            # Multiclass One-vs-Rest ROC and PR curves
            aucs, aps, roc_curves, pr_curves = [], [], [], []
            for c in range(k):
                y_bin = (y_true == c).astype(int)
                if y_bin.sum() == 0 or y_bin.sum() == n:
                    continue
                fpr, tpr, auc = _roc_curve_binary(y_bin, score_mat[:, c])
                rec_c, prec_c, ap = _pr_curve_binary(y_bin, score_mat[:, c])
                if fpr is not None:
                    aucs.append(auc)
                    roc_curves.append({"class_name": str(class_labels[c]), "fpr": fpr, "tpr": tpr, "auc": auc})
                if rec_c is not None:
                    aps.append(ap)
                    pr_curves.append({"class_name": str(class_labels[c]), "recall": rec_c, "precision": prec_c, "ap": ap, "baseline_rate": round(float(y_bin.mean()), 4)})
            if roc_curves:
                roc_data = {
                    "available": True,
                    "score_kind": score_kind,
                    "auc": round(float(np.mean(aucs)), 4),
                    "curves": roc_curves
                }
            if pr_curves:
                pr_data = {
                    "available": True,
                    "score_kind": score_kind,
                    "ap": round(float(np.mean(aps)), 4),
                    "curves": pr_curves
                }

    # Narrative interpretation of confusion matrix from actual numbers
    cm_narrative = []
    if k == 2:
        c0, c1 = per_class[0], per_class[1]
        tp, fp, fn, tn = c1["tp"], c1["fp"], c1["fn"], c1["tn"]
        cm_narrative.append(
            f"For positive class '{c1['class_name']}' vs negative class '{c0['class_name']}': "
            f"True Positives (TP) = {tp}, True Negatives (TN) = {tn}, False Positives (FP) = {fp}, False Negatives (FN) = {fn}."
        )
        if fn > fp and fn > 0:
            cm_narrative.append(f"The model misses {fn} actual '{c1['class_name']}' cases (Recall = {c1['recall']*100:.1f}%) more often than it raises false alarms ({fp} FP).")
        elif fp > fn and fp > 0:
            cm_narrative.append(f"The model produces {fp} false alarms (predicting '{c1['class_name']}' when actual is '{c0['class_name']}'), reducing Precision to {c1['precision']*100:.1f}%.")
        else:
            cm_narrative.append(f"Errors are balanced between false alarms ({fp} FP) and missed detections ({fn} FN).")
    else:
        best_cls = max(per_class, key=lambda x: x["recall"]) if per_class else None
        worst_cls = min([x for x in per_class if x["support"] > 0], key=lambda x: x["recall"], default=None)
        if best_cls and worst_cls:
            cm_narrative.append(
                f"The model identifies class '{best_cls['class_name']}' most reliably (Recall {best_cls['recall']*100:.1f}%, {best_cls['tp']}/{best_cls['support']} correct), "
                f"while struggling most on class '{worst_cls['class_name']}' (Recall {worst_cls['recall']*100:.1f}%, {worst_cls['tp']}/{worst_cls['support']} correct)."
            )
        # Find largest off-diagonal confusion pair
        off_diag = cm.copy()
        np.fill_diagonal(off_diag, 0)
        if off_diag.max() > 0:
            i_max, j_max = np.unravel_index(off_diag.argmax(), off_diag.shape)
            cm_narrative.append(
                f"Most frequent misclassification: {int(off_diag[i_max, j_max])} rows of actual class '{class_labels[i_max]}' were predicted as '{class_labels[j_max]}'."
            )

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall": round(weighted_r, 4),
        "weighted_f1": round(weighted_f1, 4),
        "roc_auc": roc_data["auc"],
        "avg_precision": pr_data["ap"],
        "confusion": cm.tolist(),
        "confusion_norm": [[round(float(v), 1) for v in row] for row in cm_norm],
        "confusion_narrative": " ".join(cm_narrative),
        "per_class": per_class,
        "roc": roc_data,
        "pr": pr_data
    }


def _regression_metrics_detail(y_true, y_pred):
    """Compute R², RMSE, MAE, and full Residual Analysis (Residual = Actual - Predicted)."""
    residuals = y_true - y_pred
    r2_val = ml.r2(y_true, y_pred)
    rmse_val = ml.rmse(y_true, y_pred)
    mae_val = ml.mae(y_true, y_pred)
    mean_res = float(residuals.mean())
    med_res = float(np.median(residuals))
    std_res = float(residuals.std())
    min_res = float(residuals.min())
    max_res = float(residuals.max())

    hist_counts, hist_edges = np.histogram(residuals, bins=10)
    rng = np.random.default_rng(1)
    pick = rng.permutation(len(y_true))[:180]
    actual_vs_pred = [[round(float(y_true[i]), 4), round(float(y_pred[i]), 4)] for i in pick]
    res_vs_pred = [[round(float(y_pred[i]), 4), round(float(residuals[i]), 4)] for i in pick]

    # Check for systematic residual patterns (correlation between |residual| and predicted = heteroscedasticity)
    het_corr = 0.0
    if np.std(y_pred) > 0 and np.std(np.abs(residuals)) > 0:
        het_corr = float(np.corrcoef(y_pred, np.abs(residuals))[0, 1])

    under_pct = float((residuals > 0).mean() * 100.0)
    over_pct = float((residuals < 0).mean() * 100.0)

    notes = [
        f"Mean residual is {mean_res:+.3f} and median residual is {med_res:+.3f} ({under_pct:.1f}% positive/underpredicted, {over_pct:.1f}% negative/overpredicted).",
        f"Average absolute error (MAE) is {mae_val:,.3f}, while RMSE is {rmse_val:,.3f} (ratio RMSE/MAE = {rmse_val/max(mae_val, 1e-9):.2f}; ratios well above 1.25 indicate large outlier errors)."
    ]
    if abs(het_corr) > 0.35:
        notes.append(f"Residual magnitude correlates with predicted value (r = {het_corr:.2f}), indicating non-constant error variance (heteroscedasticity).")
    else:
        notes.append(f"Residual spread is relatively uniform across the prediction range (r = {het_corr:.2f}).")

    return {
        "r2": round(r2_val, 4),
        "rmse": round(rmse_val, 4),
        "mae": round(mae_val, 4),
        "mean_residual": round(mean_res, 4),
        "median_residual": round(med_res, 4),
        "std_residual": round(std_res, 4),
        "min_residual": round(min_res, 4),
        "max_residual": round(max_res, 4),
        "heteroscedasticity_r": round(het_corr, 3),
        "hist_counts": hist_counts.tolist(),
        "hist_edges": [round(float(e), 3) for e in hist_edges],
        "actual_vs_pred": actual_vs_pred,
        "res_vs_pred": res_vs_pred,
        "narrative": " ".join(notes)
    }


def _bootstrap_ci(y_true, preds_by_model, task, best_name, n_boot=200):
    """Compute 95% Bootstrap Confidence Intervals on the held-out test set by resampling
    paired (y_true, y_pred) indices with replacement. Also performs paired bootstrap difference
    comparisons against the selected model."""
    n = len(y_true)
    if n < 15:
        return {
            "available": False,
            "n_test": n,
            "n_boot": 0,
            "reason": f"Held-out test set has only {n} rows (at least 15 test observations are required for reliable bootstrap resampling)."
        }

    rng = np.random.default_rng(2026)
    boot_indices = rng.integers(0, n, size=(n_boot, n))
    model_cis = {}
    boot_distributions = {}

    for m_name, y_pred in preds_by_model.items():
        if task == "classification":
            acc_samples = []
            f1_samples = []
            for idx in boot_indices:
                yb, pb = y_true[idx], y_pred[idx]
                acc_samples.append(float((yb == pb).mean()))
                f1_samples.append(ml.macro_f1(yb, pb))
            boot_distributions[m_name] = np.array(f1_samples)
            model_cis[m_name] = {
                "primary_metric": "Macro F1",
                "estimate": round(ml.macro_f1(y_true, y_pred), 4),
                "ci_low": round(float(np.percentile(f1_samples, 2.5)), 4),
                "ci_high": round(float(np.percentile(f1_samples, 97.5)), 4),
                "secondary_metric": "Accuracy",
                "sec_estimate": round(float((y_true == y_pred).mean()), 4),
                "sec_ci_low": round(float(np.percentile(acc_samples, 2.5)), 4),
                "sec_ci_high": round(float(np.percentile(acc_samples, 97.5)), 4),
            }
        else:
            r2_samples = []
            rmse_samples = []
            for idx in boot_indices:
                yb, pb = y_true[idx], y_pred[idx]
                r2_samples.append(ml.r2(yb, pb))
                rmse_samples.append(ml.rmse(yb, pb))
            boot_distributions[m_name] = np.array(r2_samples)
            model_cis[m_name] = {
                "primary_metric": "R²",
                "estimate": round(ml.r2(y_true, y_pred), 4),
                "ci_low": round(float(np.percentile(r2_samples, 2.5)), 4),
                "ci_high": round(float(np.percentile(r2_samples, 97.5)), 4),
                "secondary_metric": "RMSE",
                "sec_estimate": round(ml.rmse(y_true, y_pred), 4),
                "sec_ci_low": round(float(np.percentile(rmse_samples, 2.5)), 4),
                "sec_ci_high": round(float(np.percentile(rmse_samples, 97.5)), 4),
            }

    # Paired bootstrap comparison: Best model minus each other model on identical resamples
    paired_comparisons = []
    if best_name in boot_distributions:
        best_dist = boot_distributions[best_name]
        for m_name, dist in boot_distributions.items():
            if m_name == best_name:
                continue
            diff = best_dist - dist
            low, high = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
            mean_diff = float(np.mean(diff))
            paired_comparisons.append({
                "competitor": m_name,
                "mean_diff": round(mean_diff, 4),
                "ci_low": round(low, 4),
                "ci_high": round(high, 4),
                "statistically_clear": bool(low > 0.0 or high < 0.0),
                "interpretation": (
                    f"95% paired difference CI [{low:+.3f}, {high:+.3f}] excludes 0 (consistent advantage on this test set)."
                    if low > 0 else
                    f"95% paired difference CI [{low:+.3f}, {high:+.3f}] crosses 0 (difference is within sampling uncertainty)."
                )
            })

    return {
        "available": True,
        "n_test": n,
        "n_boot": n_boot,
        "metric_name": "Macro F1" if task == "classification" else "R²",
        "models": model_cis,
        "paired_vs_best": paired_comparisons
    }


def _learning_curves(make_model, X_dev, y_dev, task):
    """Compute leak-free Learning Curves on the 80% development set across 5 training set sizes
    (20%, 40%, 60%, 80%, 100% of training fold). Never touches the 20% held-out test set."""
    metric_fn = ml.macro_f1 if task == "classification" else ml.r2
    metric_name = "Macro F1" if task == "classification" else "R²"

    rng = np.random.default_rng(99)
    folds = ml.make_folds(y_dev, 4, rng, task == "classification")
    fractions = [0.25, 0.45, 0.65, 0.85, 1.0]
    sizes, train_means, val_means, train_stds, val_stds = [], [], [], [], []

    for frac in fractions:
        tr_scores, va_scores, n_used = [], [], 0
        for f_idx in range(min(3, len(folds))):
            val_idx = folds[f_idx]
            full_train_idx = np.setdiff1d(np.arange(len(y_dev)), val_idx)
            k = max(10, int(round(len(full_train_idx) * frac)))
            sub_rng = np.random.default_rng(100 + f_idx)
            if task == "classification":
                # Ensure all classes present in small sub-sample
                picked = []
                for c in np.unique(y_dev[full_train_idx]):
                    c_rows = full_train_idx[y_dev[full_train_idx] == c]
                    picked.append(c_rows[0])
                rem = np.setdiff1d(full_train_idx, picked)
                extra_n = max(0, min(len(rem), k - len(picked)))
                sub_idx = np.r_[picked, sub_rng.choice(rem, size=extra_n, replace=False)] if extra_n > 0 else np.array(picked)
            else:
                sub_idx = sub_rng.choice(full_train_idx, size=min(k, len(full_train_idx)), replace=False)

            n_used = len(sub_idx)
            X_tr, X_va = ml.scale(X_dev[sub_idx], X_dev[val_idx])
            m = make_model().fit(X_tr, y_dev[sub_idx])
            tr_scores.append(float(metric_fn(y_dev[sub_idx], m.predict(X_tr))))
            va_scores.append(float(metric_fn(y_dev[val_idx], m.predict(X_va))))

        sizes.append(int(n_used))
        train_means.append(round(float(np.mean(tr_scores)), 4))
        val_means.append(round(float(np.mean(va_scores)), 4))
        train_stds.append(round(float(np.std(tr_scores)), 4))
        val_stds.append(round(float(np.std(va_scores)), 4))

    final_tr = train_means[-1]
    final_va = val_means[-1]
    gap = round(final_tr - final_va, 4)
    val_gain = round(val_means[-1] - val_means[0], 4)

    if final_tr < 0.45 and final_va < 0.45:
        regime = "Underfitting (High Bias)"
        diagnosis = (
            f"Both training {metric_name} ({final_tr:.3f}) and validation {metric_name} ({final_va:.3f}) remain low even at full sample size ({sizes[-1]} rows). "
            "The model is too simple for the relationship or the available features lack predictive signal."
        )
    elif gap > 0.12:
        regime = "Overfitting (High Variance)"
        diagnosis = (
            f"Training {metric_name} ({final_tr:.3f}) is substantially higher than validation {metric_name} ({final_va:.3f}), leaving a generalization gap of {gap:+.3f}. "
            + (f"Validation performance improved by {val_gain:+.3f} as training rows increased from {sizes[0]} to {sizes[-1]}, suggesting more data or stronger regularization would help." if val_gain > 0.03 else "Adding more rows gave limited validation gain; consider simpler model settings or pruning noisy features.")
        )
    else:
        regime = "Good Generalization"
        diagnosis = (
            f"Training {metric_name} ({final_tr:.3f}) and validation {metric_name} ({final_va:.3f}) converge closely (generalization gap = {gap:+.3f}). "
            f"Moving from {sizes[0]} to {sizes[-1]} training rows changed validation {metric_name} by {val_gain:+.3f}."
        )

    return {
        "metric_name": metric_name,
        "sizes": sizes,
        "train_scores": train_means,
        "val_scores": val_means,
        "train_stds": train_stds,
        "val_stds": val_stds,
        "final_gap": gap,
        "regime": regime,
        "diagnosis": diagnosis
    }


def _build_selection_justification(task, results, per_model_test, best_item, baseline_item, bootstrap_data, is_imbalanced):
    """Generate an evidence-based, multi-criteria explanation of why the winning model was selected
    and why every alternative model was ranked lower."""
    b_name = best_item["name"]
    b_test = per_model_test[b_name]
    base_name = baseline_item["name"]
    base_test = per_model_test[base_name]

    reasons = []
    if task == "classification":
        primary_col = "Macro F1" if is_imbalanced else "Accuracy"
        b_cv_main = best_item["extra"] if is_imbalanced else best_item["score"]
        base_cv_main = baseline_item["extra"] if is_imbalanced else baseline_item["score"]
        reasons.append(
            f"Primary Selection Objective ({primary_col}): "
            + ("Because the dataset has class imbalance, Macro F1 was prioritized alongside Accuracy so minority classes are weighted equally. " if is_imbalanced else "Balanced class distribution allowed Accuracy and Macro F1 to be jointly evaluated. ")
            + f"{b_name} achieved {best_item['score']*100:.1f}% CV Accuracy (±{best_item['spread']*100:.1f}%) and {best_item['extra']:.3f} CV Macro F1, beating {base_name} by {(best_item['score']-baseline_item['score'])*100:+.1f}% Accuracy and {best_item['extra']-baseline_item['extra']:+.3f} Macro F1."
        )
        reasons.append(
            f"Held-Out Test Generalization: On the locked 20% test set, {b_name} achieved Accuracy = {b_test['accuracy']*100:.1f}%, "
            f"Macro Precision = {b_test['macro_precision']*100:.1f}%, Macro Recall = {b_test['macro_recall']*100:.1f}%, and Macro F1 = {b_test['macro_f1']:.3f}"
            + (f" (ROC-AUC = {b_test['roc_auc']:.3f})." if b_test.get("roc_auc") is not None else ".")
        )
    else:
        reasons.append(
            f"Primary Selection Objective (R² & RMSE): {b_name} achieved a cross-validation R² of {best_item['score']:.3f} (±{best_item['spread']:.3f}) "
            f"and CV RMSE of {best_item['extra']:,.3f}, improving over {base_name} (CV R² = {baseline_item['score']:.3f}, CV RMSE = {baseline_item['extra']:,.3f})."
        )
        reasons.append(
            f"Held-Out Test Generalization: On the unseen 20% test set, {b_name} achieved Test R² = {b_test['r2']:.3f}, "
            f"Test RMSE = {b_test['rmse']:,.3f}, and Test MAE = {b_test['mae']:,.3f}."
        )

    gap = best_item["train_score"] - best_item["score"]
    reasons.append(
        f"Stability & Generalization Gap: Across the 15 cross-validation folds (3 repeats × 5 folds), {b_name} ranged from {best_item['min_score']:.3f} to {best_item['max_score']:.3f} "
        f"(standard deviation = {best_item['spread']:.3f}), with a train-to-validation gap of {gap:+.3f} ({best_item['train_score']:.3f} train vs {best_item['score']:.3f} validation) and total CV runtime of {best_item['seconds']:.2f}s."
    )

    why_not_others = []
    for r in results[1:]:
        m_name = r["name"]
        m_test = per_model_test[m_name]
        if r["base"].startswith("Baseline"):
            why_not_others.append({
                "model": m_name,
                "reason": f"Naive reference model that predicts a constant value (CV {r['score']:.3f}); used strictly as a benchmark floor."
            })
            continue
        score_diff = best_item["score"] - r["score"]
        extra_diff = best_item["extra"] - r["extra"]
        r_gap = r["train_score"] - r["score"]
        parts = []
        if task == "classification":
            parts.append(f"CV Accuracy was {r['score']*100:.1f}% ({score_diff*100:+.1f}% behind {b_name}) and CV Macro F1 was {r['extra']:.3f} ({extra_diff:+.3f} difference); Test Macro F1 was {m_test['macro_f1']:.3f}.")
        else:
            parts.append(f"CV R² was {r['score']:.3f} ({score_diff:+.3f} behind {b_name}) with Test RMSE = {m_test['rmse']:,.3f} vs {b_test['rmse']:,.3f}.")
        if r["spread"] > best_item["spread"] * 1.25:
            parts.append(f"Showed higher fold-to-fold instability (±{r['spread']:.3f} vs ±{best_item['spread']:.3f}).")
        if r_gap > 0.09 and r_gap > gap + 0.03:
            parts.append(f"Exhibited stronger overfitting (train-validation gap of {r_gap:+.3f}).")
        if r["seconds"] < best_item["seconds"] * 0.5:
            parts.append(f"Although faster to train ({r['seconds']:.2f}s vs {best_item['seconds']:.2f}s), its predictive accuracy was lower.")
        if r["tie"]:
            parts.append("Note: Within 1 standard deviation of the winner (statistically close tie).")
        why_not_others.append({"model": m_name, "reason": " ".join(parts)})

    tied_models = [r["name"] for r in results[1:] if r["tie"] and not r["base"].startswith("Baseline")]
    uncertainty_note = None
    if best_item["vs_baseline"] <= 0.01 and (task != "classification" or best_item["extra_vs_baseline"] <= 0.02):
        uncertainty_note = (
            "Warning — Weak Predictive Signal: No evaluated model meaningfully outperformed the naive baseline. "
            "Do not deploy any of these models without collecting better features."
        )
    elif tied_models:
        uncertainty_note = (
            f"Statistical Tie Notice: {', '.join(tied_models)} scored within one cross-validation standard deviation (±{best_item['spread']:.3f}) "
            f"of {b_name}. Check the paired Bootstrap 95% Confidence Intervals and training times before locking in a final choice."
        )

    return {
        "winner": b_name,
        "selection_criteria": "Macro F1 + CV Stability (Imbalance-Aware)" if is_imbalanced else ("Accuracy + Macro F1 + CV Stability" if task == "classification" else "R² + RMSE + CV Stability"),
        "reasons": reasons,
        "why_not_others": why_not_others,
        "uncertainty_note": uncertainty_note
    }


def build_evaluation_centre(X_dev, y_dev, X_train, X_test, y_test, task, names, labels, results, fitted_models, best_item, baseline_item, tune, importance_all):
    """Assemble the complete Model Evaluation Centre payload for all trained models."""
    class_labels = [str(c) for c in labels] if (task == "classification" and labels is not None) else (
        [str(i) for i in range(int(y_test.max()) + 1)] if task == "classification" else []
    )

    per_model_test = {}
    preds_by_model = {}
    importance_by_model = {}
    main_metric = ml.METRICS[task][0]

    for r in results:
        m_name = r["name"]
        model = fitted_models[m_name]
        y_pred = model.predict(X_test)
        preds_by_model[m_name] = y_pred

        if task == "classification":
            score_mat, score_kind = model.predict_scores(X_test) if hasattr(model, "predict_scores") else (None, "none")
            detail = _classification_metrics_detail(y_test, y_pred, score_mat, score_kind, class_labels)
        else:
            detail = _regression_metrics_detail(y_test, y_pred)

        per_model_test[m_name] = detail

        # Permutation importance on the held-out test set for each model (3 repeats for non-best to keep fast)
        if m_name == best_item["name"]:
            importance_by_model[m_name] = importance_all
        elif not r["base"].startswith("Baseline"):
            drops = ml.permutation_importance(model, X_test, y_test, main_metric, repeats=3)
            importance_by_model[m_name] = [[names[j], round(float(drops[j]), 4)] for j in range(len(names))]
        else:
            importance_by_model[m_name] = [[names[j], 0.0] for j in range(len(names))]

    bootstrap_data = _bootstrap_ci(y_test, preds_by_model, task, best_item["name"], n_boot=200)
    learning_curve_data = _learning_curves(best_item["make"], X_dev, y_dev, task)

    is_imbalanced = (task == "classification" and float(np.bincount(y_dev).max() / len(y_dev)) > 0.70)
    selection_info = _build_selection_justification(
        task, results, per_model_test, best_item, baseline_item, bootstrap_data, is_imbalanced
    )

    return {
        "task": task,
        "class_labels": class_labels,
        "n_dev": int(len(y_dev)),
        "n_test": int(len(y_test)),
        "is_binary": (task == "classification" and len(class_labels) == 2),
        "per_model_test": per_model_test,
        "bootstrap": bootstrap_data,
        "learning_curves": learning_curve_data,
        "importance_by_model": importance_by_model,
        "selection": selection_info
    }
