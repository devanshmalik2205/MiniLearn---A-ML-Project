"""Checks a CSV before modelling, interprets columns semantically and statistically,
and executes a two-stage ML recommendation and candidate validation engine.
Strictly adheres to the university AI/ML course syllabus."""
import re
from collections import Counter

import numpy as np

import minilearn as ml

ODD_MISSING = {"n/a", "n.a.", "-", "--", "unknown", "missing", "not available", "undefined", "#n/a"}

# Semantic keyword dictionaries used ONLY as supporting evidence alongside statistical signals
ID_KEYWORDS = {"id", "uuid", "guid", "code", "number", "no", "num", "serial", "key", "index", "idx",
               "ticket", "cabin", "zip", "postal", "postcode", "phone", "tel", "ssn", "account", "empid",
               "customerid", "passengerid", "userid", "orderid", "transactionid", "recordid"}
DATE_KEYWORDS = {"date", "time", "timestamp", "datetime", "year", "month", "day", "created", "updated", "dob", "birth"}
DEMOGRAPHIC_ATTR_KEYWORDS = {"age", "gender", "sex", "marital", "siblings", "sibsp", "parch", "children",
                             "dependents", "tenure", "experience", "height", "weight", "embarked", "city", "state", "country"}
OUTCOME_CLASS_KEYWORDS = {"survived", "churn", "default", "fraud", "status", "outcome", "target", "label",
                          "diagnosis", "disease", "approved", "loan_status", "result", "passed", "class",
                          "category", "type", "tier", "grade", "satisfaction", "membership", "segment",
                          "species", "malignant", "benign", "diabetes", "cancer", "heart", "stroke"}
OUTCOME_REG_KEYWORDS = {"price", "salary", "income", "wage", "cost", "revenue", "sales", "amount", "total",
                        "score", "spending", "rating", "value", "profit", "loss", "fare", "charges",
                        "premium", "demand", "consumption", "yield", "mpg", "medv", "progression"}

DATE_PATTERNS = [
    re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"),
    re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"),
    re.compile(r"^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}"),
]


def _clean_number(v):
    return re.sub(r"[$€£%,\s]", "", v)


def _tokens(name):
    """Split column name into lowercase alphanumeric tokens."""
    clean = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    return [t for t in re.split(r"[^a-z0-9]+", clean) if t]


def _looks_like_date(present):
    if not present:
        return False
    sample = present[:50]
    matches = sum(1 for v in sample if any(p.match(v.strip()) for p in DATE_PATTERNS))
    return matches / len(sample) >= 0.7


def _looks_like_code_or_id(name, ctype, present, n_rows):
    """Detect identifiers, postal codes, serial numbers, or arbitrary reference codes."""
    if not present:
        return False, ""
    toks = set(_tokens(name))
    joined = "".join(_tokens(name))
    uniq_ratio = len(set(present)) / max(len(present), 1)

    if ctype == "numeric":
        if ml.looks_like_id(present):
            return True, "Sequential integer values (1, 2, 3...) that uniquely index each row."
        nums = [float(v) for v in present if ml.is_number(v)]
        all_int = all(x.is_integer() for x in nums)
        if all_int and len(set(nums)) == len(nums) and len(nums) >= 15:
            return True, f"Every observation has a unique integer value ({len(nums)} distinct values), acting as a unique identifier."
        if ("id" in toks or joined in ID_KEYWORDS or any(k in toks for k in ("zip", "postal", "postcode", "phone", "ssn", "serial"))) and (uniq_ratio > 0.35 or all_int):
            return True, f"Column name ('{name}') and discrete integer values indicate an identifier or reference code."
        if all_int and len(nums) >= 20 and min(nums) >= 10000 and uniq_ratio > 0.85:
            return True, f"High-uniqueness integer numbers ({uniq_ratio*100:.0f}% unique) typical of an ID or reference number."
    else:
        if uniq_ratio > 0.85 and len(present) >= 15:
            return True, f"Nearly every row has a distinct text string ({len(set(present))}/{len(present)} unique), acting as an identifier or free-text field."
        if ("id" in toks or joined in ID_KEYWORDS or "ticket" in toks or "cabin" in toks or "code" in toks or "zip" in toks or "postal" in toks) and uniq_ratio > 0.25:
            return True, f"Column name ('{name}') and high cardinality ({len(set(present))} unique values) indicate a reference code or identifier."
    return False, ""


def interpret_columns(header, rows, info):
    """Examine every column using statistical distribution, cardinality, missingness, and naming cues."""
    n = len(rows)
    interpretations = []

    for i, c in enumerate(info):
        name = c["name"]
        ctype = c["type"]
        missing = c["missing"]
        miss_pct = 100.0 * missing / max(n, 1)
        unique = c["unique"]
        present = [r[i] for r in rows if not ml.is_missing(r[i])]
        toks = set(_tokens(name))
        joined = "".join(_tokens(name))

        role = "Attribute"
        target_candidate = False
        candidate_task = None
        target_priority = 50
        notes = []

        if ctype == "empty" or len(present) == 0:
            interpretations.append({
                "name": name, "type": ctype, "unique": unique, "missing_pct": round(miss_pct, 1),
                "role": "Empty Column", "target_candidate": False, "candidate_task": None,
                "feature_usable": False,
                "explanation": "Contains 100% missing values and carries no information."
            })
            continue

        if unique < 2:
            interpretations.append({
                "name": name, "type": ctype, "unique": unique, "missing_pct": round(miss_pct, 1),
                "role": "Constant Column", "target_candidate": False, "candidate_task": None,
                "feature_usable": False,
                "explanation": f"Every non-missing row has the exact same value ('{present[0]}'). Zero variance means nothing can be learned."
            })
            continue

        # Check Temporal / Date
        if _looks_like_date(present) or (any(k in toks for k in DATE_KEYWORDS) and (ctype == "text" or (ctype == "numeric" and all(1900 <= float(v) <= 2100 for v in present[:30] if ml.is_number(v))))):
            interpretations.append({
                "name": name, "type": ctype, "unique": unique, "missing_pct": round(miss_pct, 1),
                "role": "Temporal / Date Attribute", "target_candidate": False, "candidate_task": None,
                "feature_usable": ctype == "numeric",
                "explanation": f"Represents dates or time steps ({unique} distinct values). Useful for ordering or context, not a standard cross-sectional prediction target."
            })
            continue

        # Check Identifier / Reference Code
        is_id, id_reason = _looks_like_code_or_id(name, ctype, present, n)
        if is_id:
            interpretations.append({
                "name": name, "type": ctype, "unique": unique, "missing_pct": round(miss_pct, 1),
                "role": "Identifier / Reference Code", "target_candidate": False, "candidate_task": None,
                "feature_usable": False,
                "explanation": id_reason + " Excluded from modelling to prevent memorisation."
            })
            continue

        # Evaluate Numeric Columns
        if ctype == "numeric":
            nums = np.array([float(v) for v in present if ml.is_number(v)])
            all_int = all(float(x).is_integer() for x in nums)
            val_min, val_max, val_mean, val_std = float(nums.min()), float(nums.max()), float(nums.mean()), float(nums.std())

            if unique == 2:
                # Binary numeric (e.g. 0/1 indicator or target like Survived)
                is_outcome_kw = bool(toks & OUTCOME_CLASS_KEYWORDS) or joined in OUTCOME_CLASS_KEYWORDS
                is_attr_kw = bool(toks & DEMOGRAPHIC_ATTR_KEYWORDS)
                role = "Binary Outcome / Indicator" if is_outcome_kw else ("Binary Attribute" if is_attr_kw else "Binary Category (0/1)")
                target_candidate = miss_pct <= 40.0 and not is_attr_kw
                candidate_task = "Binary Classification" if target_candidate else None
                target_priority = 90 if is_outcome_kw else (35 if is_attr_kw else 70)
                notes.append(f"Binary numeric variable taking values {{{', '.join(sorted(set(present)))}}} ({miss_pct:.0f}% missing).")
            elif unique <= 10 and all_int:
                # Small discrete integers (e.g. Pclass 1,2,3 or SibSp 0..8 or rating 1..5)
                is_count_or_attr = bool(toks & DEMOGRAPHIC_ATTR_KEYWORDS) or any(k in toks for k in ("count", "num", "sibsp", "parch", "children", "rooms", "bedrooms", "bathrooms", "floors"))
                is_outcome_kw = bool(toks & OUTCOME_CLASS_KEYWORDS) or any(k in toks for k in ("class", "grade", "tier", "status", "level", "satisfaction", "quality"))
                if is_count_or_attr and not is_outcome_kw:
                    role = "Discrete Numerical Attribute (Count)"
                    target_candidate = False
                    candidate_task = None
                    notes.append(f"Discrete count or demographic attribute ({unique} unique values from {val_min:g} to {val_max:g}). Better used as an input feature than a prediction target.")
                else:
                    role = "Ordinal / Multiclass Category"
                    target_candidate = miss_pct <= 40.0
                    candidate_task = "Multiclass Classification" if target_candidate else None
                    target_priority = 80 if is_outcome_kw else 55
                    notes.append(f"Discrete numerical category with {unique} levels ({val_min:g} to {val_max:g}).")
            else:
                # Continuous or wide-range numerical column (unique > 10 or non-integer)
                is_outcome_kw = bool(toks & OUTCOME_REG_KEYWORDS) or joined in OUTCOME_REG_KEYWORDS
                is_attr_kw = bool(toks & DEMOGRAPHIC_ATTR_KEYWORDS)
                if is_attr_kw and not is_outcome_kw:
                    role = "Continuous Explanatory Attribute"
                    # Demographic attributes like Age are primarily explanatory features, low priority as targets
                    target_candidate = False
                    candidate_task = None
                    target_priority = 25
                    notes.append(f"Explanatory numerical measurement (range {val_min:,.2f} to {val_max:,.2f}, mean {val_mean:,.2f}). Naturally acts as an input feature rather than a primary target.")
                else:
                    role = "Continuous Numerical Quantity (Target Candidate)" if is_outcome_kw else "Continuous Numerical Variable"
                    target_candidate = miss_pct <= 40.0 and unique > 10
                    candidate_task = "Regression" if target_candidate else None
                    target_priority = 90 if is_outcome_kw else 65
                    notes.append(f"Continuous numerical variable with {unique} distinct values (range {val_min:,.2f} to {val_max:,.2f}, std {val_std:,.2f}).")

            interpretations.append({
                "name": name, "type": ctype, "unique": unique, "missing_pct": round(miss_pct, 1),
                "role": role, "target_candidate": target_candidate, "candidate_task": candidate_task,
                "target_priority": target_priority,
                "feature_usable": miss_pct < 60.0,
                "explanation": " ".join(notes)
            })

        # Evaluate Text / Categorical Columns
        else:
            share_num = sum(ml.is_number(_clean_number(v)) for v in present) / max(len(present), 1)
            if share_num >= 0.8:
                interpretations.append({
                    "name": name, "type": ctype, "unique": unique, "missing_pct": round(miss_pct, 1),
                    "role": "Unparsed Numeric Text", "target_candidate": False, "candidate_task": None,
                    "feature_usable": False,
                    "explanation": "Contains numbers formatted with currency or punctuation symbols ($, %, commas). Clean formatting to use as a numeric feature."
                })
                continue

            if unique > 15:
                interpretations.append({
                    "name": name, "type": ctype, "unique": unique, "missing_pct": round(miss_pct, 1),
                    "role": "High-Cardinality Text", "target_candidate": False, "candidate_task": None,
                    "feature_usable": False,
                    "explanation": f"Contains {unique} distinct text categories across {len(present)} rows. Too many unique categories for standard one-hot encoding or classification."
                })
                continue

            is_outcome_kw = bool(toks & OUTCOME_CLASS_KEYWORDS) or joined in OUTCOME_CLASS_KEYWORDS
            is_attr_kw = bool(toks & DEMOGRAPHIC_ATTR_KEYWORDS)
            task_kind = "Binary Classification" if unique == 2 else "Multiclass Classification"

            if is_attr_kw and not is_outcome_kw:
                role = "Categorical Explanatory Attribute"
                target_candidate = False
                candidate_task = None
                target_priority = 25
                expl = f"Demographic or contextual attribute with {unique} categories. Serves as an explanatory input feature rather than a prediction target."
            else:
                role = f"Categorical Outcome ({unique} classes)" if is_outcome_kw else f"Categorical Variable ({unique} classes)"
                target_candidate = miss_pct <= 40.0 and unique <= 12
                candidate_task = task_kind if target_candidate else None
                target_priority = 90 if is_outcome_kw else (75 if i == len(info) - 1 else 58)
                expl = f"Categorical column with {unique} distinct classes ({miss_pct:.0f}% missing)."

            interpretations.append({
                "name": name, "type": ctype, "unique": unique, "missing_pct": round(miss_pct, 1),
                "role": role, "target_candidate": target_candidate, "candidate_task": candidate_task,
                "target_priority": target_priority,
                "feature_usable": miss_pct < 60.0 and unique <= 10,
                "explanation": expl
            })

    return interpretations


def _detect_leakage_and_associations(header, rows, info, interpretations, target_name, task_type):
    """Check each usable feature against target_name for target leakage and statistical association.
    Distinguishes near-exact mathematical/duplicate leakage from legitimate predictive relationships."""
    t_idx = header.index(target_name)
    valid_rows = [r for r in rows if not ml.is_missing(r[t_idx])]
    if task_type == "Regression":
        valid_rows = [r for r in valid_rows if ml.is_number(r[t_idx])]
    if len(valid_rows) < 10:
        return [], [], [], 0.0

    if task_type == "Regression":
        y_vals = np.array([float(r[t_idx]) for r in valid_rows])
    else:
        u_labels = sorted({r[t_idx] for r in valid_rows})
        y_vals = np.array([u_labels.index(r[t_idx]) for r in valid_rows])

    t_toks = set(_tokens(target_name))
    clean_features = []
    leaked_features = []
    associations = []

    for i, col_interp in enumerate(interpretations):
        fname = col_interp["name"]
        if fname == target_name or not col_interp.get("feature_usable", False):
            continue

        f_toks = set(_tokens(fname))
        ctype = info[i]["type"]
        f_vals_raw = [r[i] for r in valid_rows]
        present_mask = np.array([not ml.is_missing(v) for v in f_vals_raw])
        if present_mask.sum() < 10:
            continue

        assoc_strength = 0.0
        assoc_desc = ""
        is_leak = False
        leak_reason = ""

        if ctype == "numeric":
            f_num = np.array([float(v) if ml.is_number(v) and not ml.is_missing(v) else np.nan for v in f_vals_raw])
            mask = ~np.isnan(f_num)
            if mask.sum() < 10 or np.std(f_num[mask]) == 0:
                continue

            if task_type == "Regression":
                corr = abs(float(np.corrcoef(f_num[mask], y_vals[mask])[0, 1])) if np.std(y_vals[mask]) > 0 else 0.0
                assoc_strength = corr
                assoc_desc = f"|r| = {corr:.2f}"
                # Check if identical, constant multiple, or extreme near-1.0 correlation
                ratios = y_vals[mask] / np.where(f_num[mask] == 0, np.nan, f_num[mask])
                valid_ratios = ratios[~np.isnan(ratios)]
                is_exact_multiple = len(valid_ratios) > 5 and (np.std(valid_ratios) / (abs(np.mean(valid_ratios)) + 1e-9) < 1e-4)
                if corr >= 0.985 or is_exact_multiple:
                    is_leak = True
                    leak_reason = f"'{fname}' has a near-exact linear relationship with '{target_name}' (|r| = {corr:.3f}), indicating a duplicate, rescaled, or directly derived target column."
                elif corr >= 0.93 and (t_toks & f_toks):
                    is_leak = True
                    leak_reason = f"'{fname}' shares naming with '{target_name}' and exhibits |r| = {corr:.3f}, strongly suggesting post-outcome target leakage."
            else:
                # Classification target vs Numeric feature: compute ANOVA eta-squared (correlation ratio)
                all_f = f_num[mask]
                y_m = y_vals[mask]
                ss_tot = float(((all_f - all_f.mean()) ** 2).sum())
                if ss_tot > 0:
                    ss_btw = float(sum((y_m == c).sum() * (all_f[y_m == c].mean() - all_f.mean()) ** 2 for c in np.unique(y_m)))
                    eta = float(np.sqrt(max(0.0, ss_btw / ss_tot)))
                else:
                    eta = 0.0
                assoc_strength = eta
                assoc_desc = f"eta = {eta:.2f}"
                # Check if a simple threshold on f_num deterministically defines y_vals
                if eta >= 0.96:
                    is_leak = True
                    leak_reason = f"'{fname}' almost completely separates the classes of '{target_name}' (eta = {eta:.3f}), suggesting '{target_name}' was directly thresholded or derived from '{fname}'."
        else:
            # Categorical feature
            cats = [v if not ml.is_missing(v) else "__MISSING__" for v in f_vals_raw]
            if task_type == "Regression":
                # Eta between categorical feature and numeric target
                ss_tot = float(((y_vals - y_vals.mean()) ** 2).sum())
                if ss_tot > 0:
                    groups = {}
                    for c_val, y_v in zip(cats, y_vals):
                        if c_val != "__MISSING__":
                            groups.setdefault(c_val, []).append(y_v)
                    ss_btw = float(sum(len(g) * (np.mean(g) - y_vals.mean()) ** 2 for g in groups.values()))
                    eta = float(np.sqrt(max(0.0, ss_btw / ss_tot)))
                else:
                    eta = 0.0
                assoc_strength = eta
                assoc_desc = f"eta = {eta:.2f}"
                if eta >= 0.98:
                    is_leak = True
                    leak_reason = f"Category levels in '{fname}' deterministically map to '{target_name}' (eta = {eta:.3f})."
            else:
                # Categorical feature vs Categorical target: check purity / Cramer's V
                table = {}
                for c_val, y_v in zip(cats, y_vals):
                    if c_val == "__MISSING__":
                        continue
                    table.setdefault(c_val, Counter())[int(y_v)] += 1
                total_valid = sum(sum(cnts.values()) for cnts in table.values())
                if total_valid > 0:
                    correct_by_cat = sum(cnts.most_common(1)[0][1] for cnts in table.values())
                    purity = correct_by_cat / total_valid
                    base_majority = Counter(y_vals).most_common(1)[0][1] / len(y_vals)
                    lift = max(0.0, (purity - base_majority) / max(1.0 - base_majority, 1e-6))
                    assoc_strength = float(lift)
                    assoc_desc = f"class purity = {purity*100:.0f}%"
                    if purity >= 0.985 and len(table) >= 2:
                        is_leak = True
                        leak_reason = f"'{fname}' predicts '{target_name}' with {purity*100:.1f}% deterministic purity alone, indicating a 1-to-1 derived label or target leakage."

        if is_leak:
            leaked_features.append({"name": fname, "reason": leak_reason, "metric": assoc_desc})
        else:
            clean_features.append(fname)
            associations.append((fname, assoc_strength, assoc_desc))

    associations.sort(key=lambda x: -x[1])
    max_assoc = associations[0][1] if associations else 0.0
    return clean_features, leaked_features, associations, max_assoc


def _validate_supervised_candidate(header, rows, target_name, task_key, clean_features):
    """Run a fast, leak-free 3-fold cross-validation comparing syllabus models against Baseline.
    Uses a capped sample (up to 600 rows) for rapid validation without touching the final 20% test split."""
    drop = [h for h in header if h != target_name and h not in clean_features]
    keep_idx = [i for i, h in enumerate(header) if h not in drop]
    sub_header = [header[i] for i in keep_idx]
    sub_rows = [[r[i] for i in keep_idx] for r in rows]

    if len(sub_rows) > 600:
        pick = np.random.default_rng(42).choice(len(sub_rows), 600, replace=False)
        sub_rows = [sub_rows[i] for i in pick]

    X, y, labels, names = ml.prepare_data(sub_header, sub_rows, target_name, task_key)
    if len(names) == 0 or len(y) < 30:
        return None

    if task_key == "classification":
        if len(set(y)) < 2 or len(set(y)) > 20:
            return None
        base_res = ml.cross_validate(lambda: ml.Baseline("classification"), X, y, "classification", repeats=1, folds=3, seed=10)
        lr_res = ml.cross_validate(lambda: ml.LogisticRegression(lr=0.1, epochs=120), X, y, "classification", repeats=1, folds=3, seed=10)
        dt_res = ml.cross_validate(lambda: ml.DecisionTree("classification", max_depth=5), X, y, "classification", repeats=1, folds=3, seed=10)
        best_res, best_alg = (lr_res, "Logistic Regression") if (lr_res["score"] + lr_res["extra"]) >= (dt_res["score"] + dt_res["extra"]) else (dt_res, "Decision Tree")
        acc_gain = best_res["score"] - base_res["score"]
        f1_gain = best_res["extra"] - base_res["extra"]
        maj_share = float(np.bincount(y).max() / len(y))
        learns = (acc_gain > 0.015) or (maj_share > 0.75 and acc_gain >= -0.005 and f1_gain > 0.08)
        return {
            "metric_name": "Accuracy",
            "extra_name": "Macro F1",
            "baseline_score": round(base_res["score"], 3),
            "baseline_extra": round(base_res["extra"], 3),
            "val_score": round(best_res["score"], 3),
            "val_extra": round(best_res["extra"], 3),
            "gain": round(acc_gain, 3),
            "extra_gain": round(f1_gain, 3),
            "best_probe": best_alg,
            "learns_signal": learns,
            "summary": f"3-fold CV ({best_alg}): Accuracy {best_res['score']*100:.1f}% vs Baseline {base_res['score']*100:.1f}% ({acc_gain*100:+.1f}%), Macro F1 {best_res['extra']:.3f} vs Baseline {base_res['extra']:.3f} ({f1_gain:+.3f})."
        }
    else:
        if np.std(y) == 0:
            return None
        base_res = ml.cross_validate(lambda: ml.Baseline("regression"), X, y, "regression", repeats=1, folds=3, seed=10)
        ridge_res = ml.cross_validate(lambda: ml.RidgeRegression(1.0), X, y, "regression", repeats=1, folds=3, seed=10)
        dt_res = ml.cross_validate(lambda: ml.DecisionTree("regression", max_depth=5), X, y, "regression", repeats=1, folds=3, seed=10)
        best_res, best_alg = (ridge_res, "Ridge Regression") if ridge_res["score"] >= dt_res["score"] else (dt_res, "Decision Tree Regressor")
        r2_gain = best_res["score"] - base_res["score"]
        return {
            "metric_name": "R²",
            "extra_name": "RMSE",
            "baseline_score": round(base_res["score"], 3),
            "baseline_extra": round(base_res["extra"], 3),
            "val_score": round(best_res["score"], 3),
            "val_extra": round(best_res["extra"], 3),
            "gain": round(r2_gain, 3),
            "extra_gain": round(base_res["extra"] - best_res["extra"], 3),
            "best_probe": best_alg,
            "learns_signal": best_res["score"] > 0.05,
            "summary": f"3-fold CV ({best_alg}): R² = {best_res['score']:.3f} (Baseline R² = {base_res['score']:.3f}), RMSE = {best_res['extra']:,.2f} (Baseline RMSE = {base_res['extra']:,.2f})."
        }


def _score_label(score):
    if score >= 80:
        return "Highly suitable"
    if score >= 60:
        return "Suitable"
    if score >= 40:
        return "Needs attention"
    return "Not recommended"


def _syllabus_algorithms(task_type, n_rows, n_classes=2):
    """Return strictly syllabus-approved algorithms, explicitly distinguishing implemented vs unavailable ones."""
    if "Classification" in task_type:
        return [
            {"name": "Logistic Regression", "available": True, "reason": "Linear probabilistic baseline trained via gradient descent."},
            {"name": "Support Vector Machine (Linear SVM)", "available": True, "reason": "Maximum-margin linear classifier."},
            {"name": "Decision Tree", "available": True, "reason": "Captures non-linear feature interactions via Gini impurity splits."},
            {"name": "Decision Forest / Random Forest", "available": True, "reason": "Bagged ensemble of decision trees that reduces variance."},
            {"name": "K-Nearest Neighbours (KNN)", "available": True, "reason": "Distance-based classification on scaled features."},
            {"name": "Bayesian Learning (Gaussian Naive Bayes)", "available": True, "reason": "Applies Bayes' rule assuming conditional feature independence."},
            {"name": "Boosting / Standalone Bagging / ANN", "available": False, "reason": "In course syllabus, but not implemented in MiniLearn's lightweight engine."}
        ]
    if task_type == "Regression":
        return [
            {"name": "Linear Regression", "available": True, "reason": "Ordinary least squares linear fit."},
            {"name": "Ridge Regression", "available": True, "reason": "L2-regularised linear model that stabilises correlated features."},
            {"name": "Decision Tree Regressor", "available": True, "reason": "Piecewise-constant non-linear regression."},
            {"name": "Decision Forest / Random Forest Regressor", "available": True, "reason": "Ensemble averaging of regression trees."},
            {"name": "K-Nearest Neighbours (KNN Regressor)", "available": True, "reason": "Local neighbourhood averaging."},
            {"name": "Lasso Regression / Boosting / ANN", "available": False, "reason": "In course syllabus, but not implemented in MiniLearn's lightweight engine."}
        ]
    if task_type == "Clustering":
        return [
            {"name": "K-Means", "available": True, "reason": "Partitions observations into k compact spherical clusters around centroids (suitable for scaled numerical features)."},
            {"name": "DBSCAN", "available": False, "reason": "Syllabus algorithm for arbitrary-shaped density clusters and noise; not yet implemented in MiniLearn."},
            {"name": "BIRCH", "available": False, "reason": "Syllabus algorithm for incremental hierarchical clustering on large datasets; not yet implemented in MiniLearn."}
        ]
    if task_type == "Anomaly detection":
        return [
            {"name": "Density Estimation / Parametric Z-Score Distance", "available": True, "reason": "Estimates multivariate Gaussian parameters (MLE mean & variance) to flag low-density tail observations."},
            {"name": "Non-parametric Density-Based Anomaly Detection", "available": False, "reason": "In course syllabus, but only parametric Gaussian density distance is currently implemented."}
        ]
    if task_type == "Dimensionality Reduction":
        return [
            {"name": "Principal Component Analysis (PCA)", "available": True, "reason": "Orthogonal linear projection onto eigenvectors of the covariance matrix capturing maximum variance."},
            {"name": "Kernel PCA", "available": False, "reason": "Syllabus technique for non-linear manifold projection; not yet implemented in MiniLearn."}
        ]
    return []


def recommend_tasks(header, rows, info, interpretations=None):
    """Two-Stage Intelligent Recommendation Engine:
    Stage 1: Candidate Discovery & Semantic Filtering
    Stage 2: Leak-Free Validation, Evidence-Based Suitability Scoring & Ranking"""
    if interpretations is None:
        interpretations = interpret_columns(header, rows, info)

    n = len(rows)
    candidates = []
    rejected = []

    # Record why non-target columns were excluded in Stage 1
    for col_interp in interpretations:
        if not col_interp["target_candidate"]:
            rejected.append({
                "column": col_interp["name"],
                "role": col_interp["role"],
                "stage": "Stage 1: Candidate Discovery",
                "reason": col_interp["explanation"]
            })

    # Stage 1 & 2 for Supervised Candidates
    for i, col_interp in enumerate(interpretations):
        if not col_interp["target_candidate"]:
            continue

        target_name = col_interp["name"]
        task_type = col_interp["candidate_task"]
        task_key = "regression" if task_type == "Regression" else "classification"
        t_idx = header.index(target_name)
        valid_target_rows = [r for r in rows if not ml.is_missing(r[t_idx])]
        if task_key == "regression":
            valid_target_rows = [r for r in valid_target_rows if ml.is_number(r[t_idx])]
        n_valid = len(valid_target_rows)

        if n_valid < 30:
            rejected.append({
                "column": target_name,
                "role": col_interp["role"],
                "stage": "Stage 2: Data Sufficiency Check",
                "reason": f"Only {n_valid} rows have a valid '{target_name}' value (minimum 30 required for reliable train/validation/test splitting)."
            })
            continue

        clean_features, leaked_features, associations, max_assoc = _detect_leakage_and_associations(
            header, rows, info, interpretations, target_name, task_type
        )

        if len(clean_features) == 0:
            leak_note = f" (Excluded leaked feature(s): {', '.join(lf['name'] for lf in leaked_features)})" if leaked_features else ""
            rejected.append({
                "column": target_name,
                "role": col_interp["role"],
                "stage": "Stage 2: Feature Usefulness & Leakage Check",
                "reason": f"No usable non-leaked input features remain to predict '{target_name}'{leak_note}."
            })
            continue

        # Order suggested features by statistical association with the target
        assoc_map = {fn: (strength, desc) for fn, strength, desc in associations}
        ordered_features = sorted(clean_features, key=lambda fn: -assoc_map.get(fn, (0.0, ""))[0])

        # Perform lightweight 3-fold CV validation
        val_res = _validate_supervised_candidate(header, rows, target_name, task_key, ordered_features)
        if val_res is None:
            rejected.append({
                "column": target_name,
                "role": col_interp["role"],
                "stage": "Stage 2: Candidate Validation",
                "reason": f"Target '{target_name}' could not be validated due to insufficient usable feature encoding or degenerate class counts."
            })
            continue

        # Evidence-Based Suitability Scoring (Transparent 7-Factor Breakdown, 100 points max)
        factors = []
        concerns = []

        # 1. Data Sufficiency (0 - 15 pts)
        if n_valid >= 200:
            pts_suff = 15
            factors.append(f"+15/15 Data Sufficiency: {n_valid} valid observations provide a reliable sample size.")
        elif n_valid >= 80:
            pts_suff = 11
            factors.append(f"+11/15 Data Sufficiency: {n_valid} valid observations (moderate sample size).")
        else:
            pts_suff = 6
            factors.append(f"+6/15 Data Sufficiency: Small sample ({n_valid} valid rows); validation scores will have higher variance.")
            concerns.append(f"Small sample size ({n_valid} rows) increases the risk of overfitting.")

        # 2. Target Meaning & Suitability (0 - 20 pts)
        prio = col_interp.get("target_priority", 60)
        if prio >= 85:
            pts_targ = 20
            factors.append(f"+20/20 Target Suitability: '{target_name}' ({col_interp['role']}) is a direct outcome/response variable.")
        elif prio >= 65:
            pts_targ = 15
            factors.append(f"+15/20 Target Suitability: '{target_name}' has a well-formed distribution for {task_type.lower()}.")
        else:
            pts_targ = 9
            factors.append(f"+9/20 Target Suitability: '{target_name}' can be modelled, though it may be an auxiliary attribute.")

        # 3. Feature Usefulness & Signal (0 - 20 pts)
        top_assoc_text = ", ".join(f"{fn} ({desc})" for fn, _, desc in associations[:3]) if associations else "weak individual associations"
        if max_assoc >= 0.35:
            pts_feat = 20
            factors.append(f"+20/20 Feature Usefulness: Strong statistical signal from {len(ordered_features)} features (top: {top_assoc_text}).")
        elif max_assoc >= 0.18:
            pts_feat = 14
            factors.append(f"+14/20 Feature Usefulness: Moderate statistical relationship across {len(ordered_features)} features ({top_assoc_text}).")
        elif max_assoc >= 0.08:
            pts_feat = 7
            factors.append(f"+7/20 Feature Usefulness: Weak linear/univariate signal ({top_assoc_text}).")
            concerns.append("Input features show weak univariate statistical association with the target.")
        else:
            pts_feat = 2
            factors.append(f"+2/20 Feature Usefulness: Near-zero statistical association between available features and '{target_name}'.")
            concerns.append("Available features carry almost no statistical relationship with the target.")

        # 4. Data Quality & Completeness (0 - 15 pts)
        t_miss = col_interp["missing_pct"]
        avg_f_miss = float(np.mean([c["missing"] / max(n, 1) * 100 for c in info if c["name"] in ordered_features])) if ordered_features else 0.0
        dupes = n - len({tuple(r) for r in rows})
        pts_qual = 15
        if t_miss > 10:
            pts_qual -= 5
            concerns.append(f"Target '{target_name}' is missing in {t_miss:.1f}% of rows (those rows are excluded during training).")
        if avg_f_miss > 15:
            pts_qual -= 5
            concerns.append(f"Input features average {avg_f_miss:.1f}% missing values (imputed via median/category omission).")
        if dupes > 0.05 * n:
            pts_qual -= 3
            concerns.append(f"Dataset contains {dupes} duplicate rows ({dupes/n*100:.1f}%), which can inflate validation scores.")
        pts_qual = max(2, pts_qual)
        factors.append(f"+{pts_qual}/15 Data Quality: Target missingness {t_miss:.1f}%, average feature missingness {avg_f_miss:.1f}%, {dupes} duplicate rows.")

        # 5. Distribution / Class Balance (0 - 15 pts)
        pts_bal = 15
        dist_summary = ""
        if task_key == "classification":
            counts = Counter(r[t_idx] for r in valid_target_rows)
            maj_cls, maj_cnt = counts.most_common(1)[0]
            min_cls, min_cnt = counts.most_common()[-1]
            maj_pct = maj_cnt / n_valid * 100.0
            dist_summary = f"{len(counts)} classes; majority class '{maj_cls}' is {maj_pct:.1f}% ({maj_cnt}/{n_valid}), smallest class '{min_cnt}' rows."
            if maj_pct > 88.0 or min_cnt < 5:
                pts_bal = 3
                concerns.append(f"Severe class imbalance: '{maj_cls}' accounts for {maj_pct:.1f}% of rows (smallest class has {min_cnt} rows). Accuracy is misleading; evaluate using Macro F1, Precision, and Recall.")
            elif maj_pct > 70.0:
                pts_bal = 9
                concerns.append(f"Moderate class imbalance ({maj_pct:.1f}% in class '{maj_cls}'). Compare models against the majority-class baseline and check Macro F1.")
            factors.append(f"+{pts_bal}/15 Class Balance: {dist_summary}")
        else:
            nums = np.array([float(r[t_idx]) for r in valid_target_rows])
            most_common_pct = Counter(nums).most_common(1)[0][1] / len(nums) * 100.0
            q25, q50, q75 = np.percentile(nums, [25, 50, 75])
            skew_Note = ""
            if np.std(nums) > 0:
                skew = float(np.mean(((nums - nums.mean()) / nums.std()) ** 3))
                if abs(skew) > 2.0:
                    pts_bal -= 4
                    skew_Note = f", skewness = {skew:.2f} (heavy tail)"
                    concerns.append(f"Target '{target_name}' is strongly skewed (skewness {skew:.2f}); extreme values may dominate RMSE.")
            if most_common_pct > 40.0:
                pts_bal -= 7
                concerns.append(f"Over {most_common_pct:.0f}% of rows share the exact same '{target_name}' value.")
            pts_bal = max(3, pts_bal)
            dist_summary = f"Median {q50:,.2f} (IQR {q25:,.2f} to {q75:,.2f}{skew_Note})."
            factors.append(f"+{pts_bal}/15 Target Distribution: {dist_summary}")

        # 6. Leakage Risk Check (0 - 15 pts bonus/penalty, or warning if leaked column was auto-excluded)
        if leaked_features:
            pts_leak = -10
            for lf in leaked_features:
                concerns.append(f"Target Leakage Detected & Auto-Excluded: {lf['reason']}")
            factors.append(f"-10 Leakage Penalty: Detected {len(leaked_features)} potential leakage column(s) ({', '.join(lf['name'] for lf in leaked_features)}), which were automatically removed from the suggested feature set.")
        else:
            pts_leak = 0
            factors.append("+0 Leakage Check: No target leakage detected among candidate input features.")

        # 7. Empirical Baseline Validation (0 - 15 pts)
        if val_res["learns_signal"]:
            if (task_key == "classification" and (val_res["gain"] >= 0.05 or val_res["extra_gain"] >= 0.08)) or (task_key == "regression" and val_res["val_score"] >= 0.25):
                pts_val = 15
                factors.append(f"+15/15 Candidate Validation: Models clearly beat the naive baseline ({val_res['summary']})")
            else:
                pts_val = 10
                factors.append(f"+10/15 Candidate Validation: Modest improvement over baseline ({val_res['summary']})")
        else:
            pts_val = -15
            factors.append(f"-15/15 Candidate Validation: Models failed to beat the naive baseline in 3-fold CV ({val_res['summary']})")
            concerns.append(f"Empirical validation warning: Quick 3-fold CV did not outperform the naive baseline ({val_res['summary']}). The features may not contain enough signal to predict '{target_name}'.")

        total_score = int(max(10, min(98, pts_suff + pts_targ + pts_feat + pts_qual + pts_bal + pts_leak + pts_val)))

        # If empirical validation shows zero signal over baseline, reject the candidate
        if not val_res["learns_signal"]:
            rejected.append({
                "column": target_name,
                "role": col_interp["role"],
                "stage": "Stage 2: Empirical Validation Against Baseline",
                "reason": f"Rejected because candidate models failed to beat the naive baseline in 3-fold CV ({val_res['summary']})."
            })
            continue

        algs = _syllabus_algorithms(task_type, n_valid)
        avail_alg_names = [a["name"] for a in algs if a["available"]]

        top_feats_str = ", ".join(ordered_features[:4]) + (f" and {len(ordered_features)-4} more" if len(ordered_features) > 4 else "")
        evidence_list = [
            f"Valid observations: {n_valid} rows with non-missing '{target_name}' ({dist_summary})",
            f"Top associated features: {top_assoc_text}" if associations else f"Features evaluated: {len(ordered_features)}",
            val_res["summary"]
        ]
        if leaked_features:
            evidence_list.append(f"Excluded leakage column(s): {', '.join(lf['name'] for lf in leaked_features)}")

        candidates.append({
            "title": f"Predict {target_name} ({task_type})",
            "type": task_type,
            "goal": "predict",
            "target": target_name,
            "score": total_score,
            "suitability_label": _score_label(total_score),
            "features": ordered_features,
            "leaked_features": leaked_features,
            "algorithms": avail_alg_names,
            "algorithm_details": algs,
            "validation": val_res,
            "score_breakdown": factors,
            "explanation": {
                "what": f"Train supervised {task_type.lower()} models to predict '{target_name}' from {len(ordered_features)} explanatory features ({top_feats_str}).",
                "why": f"'{target_name}' is identified as a {col_interp['role'].lower()} with {col_interp['unique']} distinct values, and empirical 3-fold cross-validation confirms predictive signal above the naive baseline ({val_res['summary']})",
                "real_world_question": (
                    f"How accurately can we estimate the value of '{target_name}' for a new record given its { ', '.join(ordered_features[:3]) }?"
                    if task_key == "regression" else
                    f"Which category of '{target_name}' does a record belong to based on its { ', '.join(ordered_features[:3]) }?"
                ),
                "what_model_learns": (
                    f"A mathematical function mapping {len(ordered_features)} input features to continuous values of '{target_name}', minimising squared error (evaluated via R² and RMSE)."
                    if task_key == "regression" else
                    f"Decision boundaries separating the {col_interp['unique']} classes of '{target_name}' (evaluated via Accuracy and Macro F1 against a majority-class baseline)."
                ),
                "evidence": evidence_list,
                "concerns": concerns if concerns else ["No severe data quality or leakage issues detected for this task."],
                "limitations": (
                    f"Suitability score ({total_score}/100) reflects dataset structure and baseline lift, not guaranteed production accuracy. "
                    + ("Text columns with >10 categories are excluded." if any(c["type"] == "text" and c["unique"] > 10 for c in info) else "Models are constrained to syllabus linear, tree, instance, and Bayesian learners.")
                )
            }
        })

    # Sort supervised recommendations by score and keep top 3 strongest to avoid clutter
    candidates.sort(key=lambda x: x["score"], reverse=True)
    supervised_recs = candidates[:3]

    # Stage 1 & 2 for Unsupervised Learning: Clustering, Anomaly Detection, and Dimensionality Reduction (PCA)
    unsupervised_recs = []
    usable_num_cols = [
        c["name"] for i, c in enumerate(info)
        if c["type"] == "numeric"
        and interpretations[i].get("feature_usable", False)
        and interpretations[i]["role"] not in ("Identifier / Reference Code", "Constant Column", "Empty Column", "Temporal / Date Attribute")
        and c["unique"] > 2
        and (c["missing"] / max(n, 1)) < 0.35
    ]

    # 1. Clustering Candidate Evaluation
    if len(usable_num_cols) >= 2 and n >= 30:
        idx = [header.index( col_name ) for col_name in usable_num_cols]
        X_num = np.array([[float(r[j]) if ml.is_number(r[j]) and not ml.is_missing(r[j]) else np.nan for r in rows] for j in idx]).T
        X_complete = X_num[~np.isnan(X_num).any(axis=1)]
        if len(X_complete) >= 25:
            sd = X_complete.std(axis=0)
            Z = (X_complete - X_complete.mean(axis=0)) / np.where(sd == 0, 1, sd)
            rng = np.random.default_rng(0)
            k = 3
            C = Z[rng.choice(len(Z), k, replace=False)]
            for _ in range(30):
                lab = ((Z[:, None] - C[None]) ** 2).sum(axis=2).argmin(axis=1)
                C = np.array([Z[lab == j].mean(axis=0) if (lab == j).any() else C[j] for j in range(k)])
            tss = float(((Z - Z.mean(axis=0)) ** 2).sum())
            wss = float(sum(((Z[lab == j] - C[j]) ** 2).sum() for j in range(k) if (lab == j).any()))
            var_reduction = (1.0 - wss / tss) * 100.0 if tss > 0 else 0.0
            min_cluster_pct = min((lab == j).sum() for j in range(k)) / len(Z) * 100.0

            c_score = 60
            c_factors = [
                f"+20/20 Data Sufficiency: {len(X_complete)} complete numerical rows across {len(usable_num_cols)} continuous features.",
                f"+20/20 Feature Suitability: Continuous features ({', '.join(usable_num_cols[:4])}) scaled to zero mean and unit variance.",
            ]
            if var_reduction >= 35.0:
                c_score += 20
                c_factors.append(f"+20/20 Cluster Compactness: K-Means (k=3) reduces within-cluster variance by {var_reduction:.1f}% (smallest segment holds {min_cluster_pct:.1f}% of rows).")
            else:
                c_score += 8
                c_factors.append(f"+8/20 Cluster Compactness: K-Means (k=3) explains {var_reduction:.1f}% of variance across {len(usable_num_cols)} dimensions.")

            c_algs = _syllabus_algorithms("Clustering", len(X_complete))
            unsupervised_recs.append({
                "title": f"Segment Records across {', '.join(usable_num_cols[:3])}" + (f" (+{len(usable_num_cols)-3} more)" if len(usable_num_cols) > 3 else ""),
                "type": "Clustering",
                "goal": "groups",
                "target": None,
                "score": min(92, c_score),
                "suitability_label": _score_label(min(92, c_score)),
                "features": usable_num_cols,
                "leaked_features": [],
                "algorithms": [a["name"] for a in c_algs if a["available"]],
                "algorithm_details": c_algs,
                "validation": {
                    "metric_name": "Within-Cluster Variance Reduction",
                    "val_score": round(var_reduction / 100.0, 3),
                    "learns_signal": var_reduction >= 20.0,
                    "summary": f"K-Means (k=3 on {len(X_complete)} complete rows) reduces within-cluster sum of squares by {var_reduction:.1f}%."
                },
                "score_breakdown": c_factors,
                "explanation": {
                    "what": f"Partition observations into natural segments using {len(usable_num_cols)} standardized numerical attributes ({', '.join(usable_num_cols)}).",
                    "why": f"The dataset contains {len(usable_num_cols)} continuous numerical attributes with {len(X_complete)} complete rows, and k=3 K-Means reduces within-cluster variance by {var_reduction:.1f}%.",
                    "real_world_question": f"Do records naturally group into distinct profiles based on {', '.join(usable_num_cols[:3])}?",
                    "what_model_learns": "Cluster centroids in standardized Euclidean space that minimise the within-cluster sum of squared distances.",
                    "evidence": [
                        f"{len(X_complete)} complete rows across {len(usable_num_cols)} numerical features ({', '.join(usable_num_cols)}).",
                        f"K-Means (k=3) within-cluster variance reduction: {var_reduction:.1f}% (smallest group size: {min_cluster_pct:.1f}% of data)."
                    ],
                    "concerns": [
                        "K-Means assumes convex, roughly spherical clusters and is sensitive to outliers.",
                        "DBSCAN (for irregular density clusters/noise) and BIRCH (for large-scale incremental clustering) are part of the course syllabus but are not currently implemented in MiniLearn."
                    ],
                    "limitations": "Unsupervised clustering has no ground-truth target labels; cluster profiles must be inspected to verify domain meaning."
                }
            })

    # 2. Dimensionality Reduction (PCA) Supporting Recommendation when >= 3 numeric features and correlation exists
    if len(usable_num_cols) >= 3 and n >= 20:
        idx = [header.index(col_name) for col_name in usable_num_cols]
        X_num = np.array([[float(r[j]) if ml.is_number(r[j]) and not ml.is_missing(r[j]) else np.nan for r in rows] for j in idx]).T
        col_means = np.nanmean(X_num, axis=0)
        inds = np.where(np.isnan(X_num))
        X_num[inds] = np.take(col_means, inds[1])
        sd = X_num.std(axis=0)
        Z = (X_num - X_num.mean(axis=0)) / np.where(sd == 0, 1, sd)
        cov = np.cov(Z, rowvar=False)
        eigvals = np.maximum(np.linalg.eigvalsh(cov)[::-1], 0.0)
        var_2d = float(eigvals[:2].sum() / max(eigvals.sum(), 1e-9)) * 100.0

        # Find highest pairwise correlation among usable_num_cols
        corr_mat = np.corrcoef(Z, rowvar=False)
        max_r, pair_names = 0.0, ("", "")
        for a in range(len(usable_num_cols)):
            for b in range(a + 1, len(usable_num_cols)):
                r_val = abs(float(corr_mat[a, b]))
                if r_val > max_r:
                    max_r, pair_names = r_val, (usable_num_cols[a], usable_num_cols[b])

        if max_r >= 0.30 or len(usable_num_cols) >= 4:
            pca_score = 78 if max_r >= 0.45 else 68
            p_algs = _syllabus_algorithms("Dimensionality Reduction", n)
            unsupervised_recs.append({
                "title": f"Principal Component Analysis (PCA) on {len(usable_num_cols)} Features",
                "type": "Dimensionality Reduction (Supporting Analysis)",
                "goal": "pca",
                "target": None,
                "score": pca_score,
                "suitability_label": _score_label(pca_score),
                "features": usable_num_cols,
                "leaked_features": [],
                "algorithms": [a["name"] for a in p_algs if a["available"]],
                "algorithm_details": p_algs,
                "validation": {
                    "metric_name": "2-Component Explained Variance",
                    "val_score": round(var_2d / 100.0, 3),
                    "learns_signal": True,
                    "summary": f"First 2 Principal Components capture {var_2d:.1f}% of total variance across {len(usable_num_cols)} features (strongest pair: {pair_names[0]} & {pair_names[1]}, |r| = {max_r:.2f})."
                },
                "score_breakdown": [
                    f"+30/30 Feature Redundancy: Strongest correlation is |r| = {max_r:.2f} between '{pair_names[0]}' and '{pair_names[1]}'.",
                    f"+{pca_score-30}/70 Variance Compression: PC1 + PC2 explain {var_2d:.1f}% of the variance in {len(usable_num_cols)} numerical columns."
                ],
                "explanation": {
                    "what": f"Compress {len(usable_num_cols)} correlated numerical features ({', '.join(usable_num_cols)}) into orthogonal Principal Components (supporting analysis, not a supervised prediction task).",
                    "why": f"Features share correlated variance (e.g., '{pair_names[0]}' and '{pair_names[1]}' have |r| = {max_r:.2f}), allowing the top 2 linear components to summarize {var_2d:.1f}% of total variance.",
                    "real_world_question": f"Can we summarize and visualize the {len(usable_num_cols)} numerical measurements in 2D while removing multicollinearity?",
                    "what_model_learns": "Eigenvectors (principal axes) of the standardized feature covariance matrix that capture maximum variance.",
                    "evidence": [
                        f"Top 2 Principal Components retain {var_2d:.1f}% of total variance across {len(usable_num_cols)} features.",
                        f"Max pairwise correlation: |r| = {max_r:.2f} ({pair_names[0]} vs {pair_names[1]})."
                    ],
                    "concerns": [
                        "Linear PCA captures only linear correlations. Kernel PCA (in syllabus for non-linear manifolds) is not currently implemented.",
                        "PCA is an unsupervised dimensionality reduction technique, not a target prediction model."
                    ],
                    "limitations": "Principal components are linear combinations of all input features, which are less directly interpretable than original units."
                }
            })

    # 3. Density-Based / Parametric Anomaly Detection
    if len(usable_num_cols) >= 1 and n >= 20:
        idx = [header.index(col_name) for col_name in usable_num_cols]
        X_num = np.array([[float(r[j]) if ml.is_number(r[j]) and not ml.is_missing(r[j]) else np.nan for r in rows] for j in idx]).T
        col_means = np.nanmean(X_num, axis=0)
        inds = np.where(np.isnan(X_num))
        X_num[inds] = np.take(col_means, inds[1])
        sd = X_num.std(axis=0)
        Z = (X_num - X_num.mean(axis=0)) / np.where(sd == 0, 1, sd)
        scores = (Z ** 2).sum(axis=1)
        extreme_cnt = int((np.abs(Z).max(axis=1) > 3.0).sum())
        max_z_score = float(scores.max())

        if extreme_cnt > 0:
            a_score = 74 if extreme_cnt >= 3 else 64
            a_algs = _syllabus_algorithms("Anomaly detection", n)
            unsupervised_recs.append({
                "title": f"Detect Unusual Records across {', '.join(usable_num_cols[:3])}",
                "type": "Anomaly detection",
                "goal": "anomalies",
                "target": None,
                "score": a_score,
                "suitability_label": _score_label(a_score),
                "features": usable_num_cols,
                "leaked_features": [],
                "algorithms": [a["name"] for a in a_algs if a["available"]],
                "algorithm_details": a_algs,
                "validation": {
                    "metric_name": "Extreme Tail Observations (|z| > 3)",
                    "val_score": extreme_cnt,
                    "learns_signal": extreme_cnt > 0,
                    "summary": f"Found {extreme_cnt} record(s) exceeding 3 standard deviations from the mean (peak multivariate Z² score = {max_z_score:.1f})."
                },
                "score_breakdown": [
                    f"+40/50 Numerical Coverage: Evaluated {len(usable_num_cols)} continuous features across {n} records.",
                    f"+{a_score-40}/50 Tail Evidence: {extreme_cnt} observation(s) lie beyond |z| > 3.0 (low Gaussian density region)."
                ],
                "explanation": {
                    "what": f"Identify records that fall in low-density regions across {', '.join(usable_num_cols)} using parametric Gaussian density estimation (MLE mean and variance).",
                    "why": f"{extreme_cnt} row(s) exhibit values more than 3 standard deviations away from column means (maximum Z² distance = {max_z_score:.1f}), warranting outlier inspection.",
                    "real_world_question": f"Which specific rows have extreme or anomalous combinations of {', '.join(usable_num_cols[:3])} that may represent data errors or rare cases?",
                    "what_model_learns": "Maximum Likelihood Estimates (mean and variance) for each numerical feature to compute squared standardized distance (Z²).",
                    "evidence": [
                        f"{extreme_cnt} row(s) exceed |z| > 3.0 across {len(usable_num_cols)} numerical features.",
                        f"Highest anomaly distance score: Z² = {max_z_score:.1f}."
                    ],
                    "concerns": [
                        "Parametric Z-score distance assumes roughly bell-shaped feature distributions; heavy skewness can flag natural tail values."
                    ],
                    "limitations": "High anomaly scores highlight statistical rarity, not confirmed fraud or error—domain review is required."
                }
            })

    all_recs = supervised_recs + unsupervised_recs
    return all_recs, rejected


def check(header, rows, info, notes=()):
    out = list(notes)
    n = len(rows)
    add = lambda level, title, detail: out.append(dict(level=level, title=title, detail=detail))

    if n < 30:
        add("bad", f"Only {n} rows", "Supervised model comparison needs at least 30 rows with a target value to split into training, cross-validation, and final test sets.")
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
        pct = 100 * c["missing"] / max(n, 1)
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
    """Comprehensive Dataset Understanding, Column Interpretations, Syllabus Scope Notes,
    and Validated ML Recommendations."""
    n = len(rows)
    num = [c for c in info if c["type"] == "numeric"]
    txt = [c for c in info if c["type"] == "text"]
    names = lambda cs, k=6: ", ".join(c["name"] for c in cs[:k]) + (f" and {len(cs) - k} more" if len(cs) > k else "")
    lines = [
        f"Each of the {n} rows is one observation, described by {len(header)} columns: {len(num)} numeric ({names(num)}) and {len(txt)} text ({names(txt)})."
        if txt else f"Each of the {n} rows is one observation, described by {len(header)} numeric columns ({names(num)})."
    ]
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
    lines.append(f"{len(gaps)} column(s) have missing values; the emptiest is '{gaps[0]['name']}' ({100 * gaps[0]['missing'] / max(n, 1):.0f}% blank)." if gaps else "There are no missing values across any column.")

    interpretations = interpret_columns(header, rows, info)
    suggested, rejected = recommend_tasks(header, rows, info, interpretations)

    no_task_reason = None
    if not suggested:
        if n < 30:
            no_task_reason = f"The dataset contains only {n} rows. At least 30 valid observations are required to split data into training, validation, and testing sets without severe variance."
        else:
            no_task_reason = "After examining all columns for identifiers, constants, missingness, feature associations, and empirical cross-validation against a naive baseline, no technically justified supervised or unsupervised ML task was found."

    syllabus_notes = {
        "implemented": [
            "Supervised Classification: Logistic Regression, Support Vector Machine (Linear SVM), Decision Tree, Decision Forest / Random Forest, K-Nearest Neighbours (KNN), Bayesian Learning (Gaussian Naive Bayes)",
            "Supervised Regression: Linear Regression, Ridge Regression, Decision Tree Regressor, Decision Forest / Random Forest Regressor, KNN Regressor",
            "Unsupervised & Supporting: K-Means Clustering, Density Estimation / Parametric Z-Score Anomaly Detection, Principal Component Analysis (PCA)",
            "Evaluation & Optimization: Train/Test Split, Stratified Cross-Validation, Hyperparameter Tuning, Accuracy, Recall, Macro F1-Score, R², RMSE, Permutation Feature Importance"
        ],
        "unavailable_or_separate": [
            "Lasso Regression, standalone Bagging/Boosting, DBSCAN, BIRCH, and Kernel PCA are part of the course syllabus but are not yet implemented in MiniLearn's lightweight engine (explicitly marked on recommendation cards).",
            "Artificial Neural Networks (ANN): Reserved for larger/complex datasets where multi-layer representation learning is justified; not implemented in this tabular numpy engine.",
            "Reinforcement Learning (RL): Not applicable to static CSV tabular datasets; RL requires an interactive Environment with States, Actions, Rewards, and a Policy."
        ]
    }

    return dict(
        summary=lines,
        column_interpretations=interpretations,
        suggested=suggested,
        rejected_candidates=rejected,
        no_task_reason=no_task_reason,
        syllabus_notes=syllabus_notes
    )
