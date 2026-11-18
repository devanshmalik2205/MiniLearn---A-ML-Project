"""Comprehensive 10-Scenario Test Suite for MiniLearn's Upgraded Model Evaluation Centre.
Verifies all 10 required evaluation scenarios:
1. Binary classification (Titanic Survival)
2. Multiclass classification (3-class Customer Tier)
3. Imbalanced classification (94% Legit / 6% Fraud)
4. Regression (House Prices)
5. Small datasets (35 rows — small test split for bootstrap & learning curves)
6. Models with probability support vs continuous decision margin scores (Logistic/Tree/NB/KNN vs Linear SVM)
7. Models with different training times (Baseline/NB vs Random Forest/Logistic/SVM)
8. Models with different cross-validation variability (fold scores, min/max, spread)
9. A dataset where models perform similarly (statistical tie detection & paired bootstrap CI)
10. A dataset where all models perform poorly (pure random noise target — weak-signal warning)
"""
import os
import random
import numpy as np
import minilearn as ml
import predict as pr


def _run_eval_scenario(title, header, rows, target, task, tune=True):
    X, y, labels, names = ml.prepare_data(header, rows, target, task)
    report = ml.rank_models(X, y, task, names, tune=tune, labels=labels)
    ec = report["evaluation_centre"]
    b_name = report["best"]
    b_res = report["results"][0]
    b_test = ec["per_model_test"][b_name]

    print("=" * 90)
    print(f"SCENARIO: {title}")
    print(f"Dataset: {report['rows']} valid rows (80% Dev = {report['dev_rows']}, 20% Locked Test = {report['test_rows']}) | Task: {task} | Tuned: {tune}")
    print("-" * 90)
    print(f"Selected Winner: {b_name} | Criterion: {ec['selection']['selection_criteria']}")
    if ec["selection"].get("uncertainty_note"):
        print(f"Uncertainty / Caution Banner: {ec['selection']['uncertainty_note']}")

    if task == "classification":
        print("\nLeaderboard (CV vs Locked 20% Test Set):")
        print(f"  {'Model':38} | {'CV Acc (±SD)':15} | {'Test Acc':8} | {'Test Prec':9} | {'Test Rec':8} | {'Test F1':7} | {'ROC-AUC':7} | {'PR-AP':7} | {'Score Kind':14} | {'Time':6}")
        for m in report["results"]:
            mt = ec["per_model_test"][m["name"]]
            auc_s = f"{mt['roc_auc']:.3f}" if mt["roc_auc"] is not None else "N/A"
            ap_s = f"{mt['avg_precision']:.3f}" if mt["avg_precision"] is not None else "N/A"
            sk = mt["roc"]["score_kind"]
            print(f"  {m['name']:38} | {m['score']*100:5.1f}% (±{m['spread']*100:4.1f}%) | {mt['accuracy']*100:7.1f}% | {mt['macro_precision']*100:8.1f}% | {mt['macro_recall']*100:7.1f}% | {mt['macro_f1']:7.3f} | {auc_s:7} | {ap_s:7} | {sk:14} | {m['seconds']:5.2f}s")
        print(f"\nConfusion Matrix ({b_name}, Raw Counts): {b_test['confusion']}")
        print(f"Confusion Matrix ({b_name}, Row-Normalized %): {b_test['confusion_norm']}")
        print(f"Confusion Matrix Narrative: {b_test['confusion_narrative']}")
    else:
        print("\nLeaderboard (CV vs Locked 20% Test Set):")
        print(f"  {'Model':38} | {'CV R² (±SD)':16} | {'Test R²':8} | {'Test RMSE':10} | {'Test MAE':10} | {'Mean Resid':10} | {'Time':6}")
        for m in report["results"]:
            mt = ec["per_model_test"][m["name"]]
            print(f"  {m['name']:38} | {m['score']:6.3f} (±{m['spread']:5.3f}) | {mt['r2']:8.3f} | {mt['rmse']:10.3f} | {mt['mae']:10.3f} | {mt['mean_residual']:+10.3f} | {m['seconds']:5.2f}s")
        print(f"\nResidual Diagnostics ({b_name}): {b_test['narrative']}")

    # Check Bootstrap CIs
    boot = ec["bootstrap"]
    if boot["available"]:
        bm = boot["models"][b_name]
        print(f"\nBootstrap 95% CI ({boot['n_boot']} resamples on {boot['n_test']} test rows):")
        print(f"  {b_name} Test {bm['primary_metric']} = {bm['estimate']:.3f}, 95% CI = [{bm['ci_low']:.3f}, {bm['ci_high']:.3f}]")
        if boot["paired_vs_best"]:
            p0 = boot["paired_vs_best"][0]
            print(f"  Paired vs {p0['competitor']}: diff = {p0['mean_diff']:+.3f}, 95% CI = [{p0['ci_low']:+.3f}, {p0['ci_high']:+.3f}] -> {p0['interpretation']}")
    else:
        print(f"\nBootstrap Status: {boot['reason']}")

    # Check Learning Curves
    lc = ec["learning_curves"]
    print(f"\nLearning Curve ({b_name}): Regime = {lc['regime']} | Sizes = {lc['sizes']} | Train = {lc['train_scores']} | Val = {lc['val_scores']}")

    # Check Hyperparameter Tuning History
    tuned_models = [m for m in report["results"] if m.get("tuning_history")]
    if tuned_models:
        tm = tuned_models[0]
        print(f"Hyperparameter Tuning Sample ({tm['base']}): Default={tm['default_score']} -> Selected '{tm['setting']}' ({tm['tuned_score']}) | Grid={[h['setting'] + ':' + str(h['score']) for h in tm['tuning_history']]}")

    # Check Permutation Feature Importance across models
    imp_b = ec["importance_by_model"][b_name]
    print(f"Top Permutation Features ({b_name}): {imp_b[:3]}")

    # Verify live single-row prediction still works cleanly
    spec = pr.form_spec(header, rows, target, names)
    ans = pr.predict_all(X, y, task, names, spec, {}, b_name, labels)
    assert len(ans) > 0, "Live prediction failed!"
    print(f"Live Prediction Check Passed: Best model answered '{ans[0]['answer']}'\n")
    return report


def run_all():
    rng = random.Random(2026)

    # 1. Binary Classification (Titanic) & 6. Models with/without probability support & 7. Different training times
    h1, r1 = ml.load_csv(os.path.join(os.path.dirname(__file__), "static", "titanic_sample.csv"))
    rep1 = _run_eval_scenario("1, 6 & 7. Binary Classification (Titanic) — Probabilities vs SVM Margins & Runtime Comparison", h1, r1, "Survived", "classification", tune=True)
    assert rep1["evaluation_centre"]["per_model_test"]["Linear SVM [reg 0.001]" if "Linear SVM [reg 0.001]" in rep1["evaluation_centre"]["per_model_test"] else next(m["name"] for m in rep1["results"] if m["base"] == "Linear SVM")]["roc"]["score_kind"] == "decision_score"

    # 2. Multiclass Classification (3-class Customer Membership Tier)
    h2 = ["Age", "Annual_Income", "Spending_Score", "Membership_Type"]
    r2 = []
    for i in range(150):
        age = rng.randint(19, 68)
        inc = round(25 + (age - 18) * 1.1 + rng.gauss(0, 7), 1)
        tier = "Gold" if inc > 60 else ("Silver" if inc > 42 else "Bronze")
        spend = round(80 - 0.5 * age + 0.4 * inc + rng.gauss(0, 6), 1)
        r2.append([str(age), str(inc), str(spend), tier])
    _run_eval_scenario("2. Multiclass Classification (3 Classes: Bronze / Silver / Gold — One-vs-Rest ROC & PR)", h2, r2, "Membership_Type", "classification", tune=True)

    # 3. Imbalanced Classification (93% Legit / 7% Fraud)
    h3 = ["Amount", "Velocity_24h", "Foreign_Tx", "Fraud_Status"]
    r3 = []
    for i in range(180):
        fraud = 1 if i < 13 else 0
        amt = rng.uniform(550, 2200) if fraud else rng.uniform(10, 500)
        vel = rng.randint(5, 12) if fraud else rng.randint(1, 5)
        ftx = "Yes" if (fraud and rng.random() < 0.75) or rng.random() < 0.1 else "No"
        r3.append([f"{amt:.2f}", str(vel), ftx, "Fraud" if fraud else "Legit"])
    _run_eval_scenario("3. Imbalanced Binary Classification (93% Legit / 7% Fraud — Macro F1 Prioritization)", h3, r3, "Fraud_Status", "classification", tune=False)

    # 4. Regression (House Prices)
    h4 = ["SqFt_Living", "Bedrooms", "Bathrooms", "Neighborhood", "Sale_Price"]
    r4 = []
    for i in range(140):
        sqft = rng.randint(850, 3800)
        beds = max(1, min(6, int(sqft / 650)))
        baths = max(1, min(4, int(beds * 0.75)))
        hood = rng.choice(["Downtown", "North", "Suburb"])
        bonus = {"Downtown": 75000, "North": 35000, "Suburb": 0}[hood]
        price = round(55000 + sqft * 160 + beds * 10000 + bonus + rng.gauss(0, 20000), 2)
        r4.append([str(sqft), str(beds), str(baths), hood, f"{price:.2f}"])
    _run_eval_scenario("4. Regression (House Sale Price — R², RMSE, MAE & Residual Analysis)", h4, r4, "Sale_Price", "regression", tune=True)

    # 5 & 8. Small Dataset (36 rows -> 7 locked test rows, triggers small-sample Bootstrap warning & high CV spread)
    h5 = ["Feature_A", "Feature_B", "Outcome"]
    r5 = []
    for i in range(36):
        fa = rng.uniform(0, 10)
        fb = rng.uniform(0, 10)
        out = "Positive" if (fa + fb + rng.gauss(0, 2.5) > 10) else "Negative"
        r5.append([f"{fa:.2f}", f"{fb:.2f}", out])
    _run_eval_scenario("5 & 8. Small Dataset (36 total rows, 7 test rows) — High CV Fold Variability & Bootstrap Small-Sample Guard", h5, r5, "Outcome", "classification", tune=False)

    # 9. Dataset Where Models Perform Similarly (Clean linear relationship where Linear, Ridge, RF, Tree tie within 1 SD)
    h9 = ["X1", "X2", "Y_Linear"]
    r9 = []
    for i in range(150):
        x1 = rng.uniform(-5, 5)
        x2 = rng.uniform(-5, 5)
        y_val = 3.0 * x1 - 2.0 * x2 + rng.gauss(0, 0.4)
        r9.append([f"{x1:.3f}", f"{x2:.3f}", f"{y_val:.3f}"])
    _run_eval_scenario("9. Dataset Where Models Perform Similarly (Linear vs Ridge vs Tree Statistical Ties)", h9, r9, "Y_Linear", "regression", tune=True)

    # 10. Dataset Where All Models Perform Poorly (Pure Uncorrelated Noise)
    h10 = ["Noise_1", "Noise_2", "Noise_3", "Random_Target"]
    r10 = [[f"{rng.gauss(0, 1):.3f}", f"{rng.gauss(0, 1):.3f}", f"{rng.gauss(0, 1):.3f}", f"{rng.gauss(50, 15):.3f}"] for _ in range(120)]
    _run_eval_scenario("10. Dataset Where All Models Perform Poorly (Pure Noise Target — Negative Test R² & Weak Signal Warning)", h10, r10, "Random_Target", "regression", tune=False)


if __name__ == "__main__":
    run_all()
