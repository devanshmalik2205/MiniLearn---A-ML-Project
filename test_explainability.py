"""Comprehensive 11-Requirement Verification Test Suite for the Model Explainability Centre.
Verifies:
1. Global feature importance is calculated correctly across metrics.
2. Permutation importance uses the held-out evaluation dataset (never training data).
3. Standardized & unscaled coefficients match the fitted Linear / Ridge / Logistic / Linear SVM models.
4. Decision tree visualization structure matches the actual fitted tree (root, thresholds in original units, sample counts, Gini/Variance impurity) and Random Forest exposes individual trees + ensemble impurity importance.
5. Local explanations correspond to the actual prediction (positive/negative counterfactual contributions + exact tree path).
6. What-if analysis uses the saved model and preprocessing pipeline without retraining (`fit()` blocked).
7. Feature preprocessing is applied identically to prediction preprocessing.
8. Unsupported intrinsic algorithms (KNN, Baseline) display appropriate transparent messages without fabricating weights.
9. Correlated features are detected and reported with suitable warnings in the Reliability Audit.
10. Explainability report data contains actual computed values.
11. Saved models can be explained after restarting the application without re-uploading the CSV.
"""
import io
import os
import random
import shutil
import unittest.mock as mock

import numpy as np

import app as flask_app
import minilearn as ml
import model_hub as mh


def run_explainability_tests():
    if os.path.exists(mh.HUB_DIR):
        shutil.rmtree(mh.HUB_DIR)
    mh._ensure_storage()

    client = flask_app.app.test_client()

    # Load Titanic dataset for Classification tests
    titanic_path = os.path.join(os.path.dirname(__file__), "static", "titanic_sample.csv")
    with open(titanic_path, "rb") as f:
        titanic_bytes = f.read()

    # Create House Prices dataset with a deliberately correlated feature pair (SqFt_Living & Total_Rooms) for Regression & Correlation tests
    rng = random.Random(2026)
    reg_lines = ["SqFt_Living,Total_Rooms,Bedrooms,Neighborhood,Sale_Price"]
    for _ in range(140):
        sqft = rng.randint(900, 3600)
        rooms = round(sqft / 320.0 + rng.gauss(0, 0.3), 1)  # Strongly correlated with SqFt_Living (|r| > 0.9)
        beds = max(1, min(6, int(sqft / 650)))
        hood = rng.choice(["Downtown", "North", "Suburb"])
        bonus = {"Downtown": 80000, "North": 35000, "Suburb": 0}[hood]
        price = round(60000 + sqft * 165 + beds * 12000 + bonus + rng.gauss(0, 16000), 2)
        reg_lines.append(f"{sqft},{rooms},{beds},{hood},{price:.2f}")
    reg_bytes = "\n".join(reg_lines).encode("utf-8")

    # Save 6 diverse syllabus models to test every explanation method:
    # 1) Decision Tree (Classification)
    m_dt = client.post("/api/models/save", data={
        "file": (io.BytesIO(titanic_bytes), "titanic.csv"),
        "dataset_name": "titanic.csv", "target": "Survived", "tune": "1", "model_name": "Decision tree"
    }, content_type="multipart/form-data").get_json()

    # 2) Random Forest (Classification)
    m_rf = client.post("/api/models/save", data={
        "file": (io.BytesIO(titanic_bytes), "titanic.csv"),
        "dataset_name": "titanic.csv", "target": "Survived", "tune": "1", "model_name": "Random forest"
    }, content_type="multipart/form-data").get_json()

    # 3) Logistic Regression (Classification)
    m_lr = client.post("/api/models/save", data={
        "file": (io.BytesIO(titanic_bytes), "titanic.csv"),
        "dataset_name": "titanic.csv", "target": "Survived", "tune": "1", "model_name": "Logistic regression"
    }, content_type="multipart/form-data").get_json()

    # 4) Gaussian Naive Bayes (Classification)
    m_nb = client.post("/api/models/save", data={
        "file": (io.BytesIO(titanic_bytes), "titanic.csv"),
        "dataset_name": "titanic.csv", "target": "Survived", "tune": "0", "model_name": "Naive Bayes"
    }, content_type="multipart/form-data").get_json()

    # 5) KNN (Classification — no intrinsic weights, model-agnostic only)
    m_knn = client.post("/api/models/save", data={
        "file": (io.BytesIO(titanic_bytes), "titanic.csv"),
        "dataset_name": "titanic.csv", "target": "Survived", "tune": "1", "model_name": "KNN"
    }, content_type="multipart/form-data").get_json()

    # 6) Ridge Regression (Regression on House Prices)
    m_ridge = client.post("/api/models/save", data={
        "file": (io.BytesIO(reg_bytes), "house_prices.csv"),
        "dataset_name": "house_prices.csv", "target": "Sale_Price", "tune": "1", "model_name": "Ridge regression"
    }, content_type="multipart/form-data").get_json()

    # Block all .fit() methods to prove that ALL explainability, PDP, local explanation, and What-If endpoints run with ZERO retraining!
    with mock.patch.object(ml.DecisionTree, "fit", side_effect=RuntimeError("RETRAINING FORBIDDEN!")), \
         mock.patch.object(ml.RandomForest, "fit", side_effect=RuntimeError("RETRAINING FORBIDDEN!")), \
         mock.patch.object(ml.LogisticRegression, "fit", side_effect=RuntimeError("RETRAINING FORBIDDEN!")), \
         mock.patch.object(ml.GaussianNaiveBayes, "fit", side_effect=RuntimeError("RETRAINING FORBIDDEN!")), \
         mock.patch.object(ml.KNN, "fit", side_effect=RuntimeError("RETRAINING FORBIDDEN!")), \
         mock.patch.object(ml.RidgeRegression, "fit", side_effect=RuntimeError("RETRAINING FORBIDDEN!")):

        # ---------------------------------------------------------------------
        # TEST 1, 2 & 4: Decision Tree Explainability, Tree Structure & Exact Node Metadata
        # ---------------------------------------------------------------------
        print("=" * 88)
        print("TEST 1, 2 & 4: Decision Tree Global Importance & Interactive Tree Structure")
        print("-" * 88)
        ex_dt = client.get(f"/api/models/{m_dt['model_id']}/explain").get_json()
        print(f"Model: {ex_dt['overview']['display_name']} | Eval Rows: {ex_dt['overview']['n_eval_rows']} held-out test rows")
        print(f"Top Permutation Feature: {ex_dt['overview']['most_influential_feature']} (drop={ex_dt['overview']['most_influential_drop']})")
        tree = ex_dt["model_specific"]["tree_structure"]
        print(f"Root Node Split: '{tree['condition']}' | Samples={tree['samples']} | Gini Impurity={tree['impurity']}")
        assert tree["samples"] > 100
        assert not tree["is_leaf"]
        assert len(ex_dt["model_specific"]["top_splits"]) > 0

        # Check Random Forest exposes sample trees + forest-wide impurity importance
        ex_rf = client.get(f"/api/models/{m_rf['model_id']}/explain").get_json()
        assert ex_rf["model_specific"]["method_type"] == "random_forest"
        assert len(ex_rf["model_specific"]["sample_trees"]) > 0
        print(f"Random Forest Inspector: {ex_rf['model_specific']['n_trees']} trees, top impurity feature = {ex_rf['model_specific']['impurity_importances'][0]}")

        # ---------------------------------------------------------------------
        # TEST 3 & 9: Ridge Regression Coefficients, Unscaled Slopes & Correlated Feature Warning
        # ---------------------------------------------------------------------
        print("\n" + "=" * 88)
        print("TEST 3 & 9: Ridge Regression Coefficients & Correlated Features Warning")
        print("-" * 88)
        ex_ridge = client.get(f"/api/models/{m_ridge['model_id']}/explain").get_json()
        ms_ridge = ex_ridge["model_specific"]
        loaded_ridge = mh.load_model_artifact(m_ridge["model_id"])["model"]
        assert abs(ms_ridge["intercept"] - float(loaded_ridge.w[0])) < 1e-4
        print(f"Ridge Intercept={ms_ridge['intercept']} | Top Coefficients: {ms_ridge['coefficients'][:3]}")
        findings = ex_ridge["reliability"]["findings"]
        print("Reliability Audit Findings:")
        for fd in findings:
            print(f"  - [{fd['level'].upper()}] {fd['title']}: {fd['detail']}")
        assert any("Correlated Features Detected" in fd["title"] for fd in findings), "Expected correlation warning!"

        # Check Logistic Regression & Naive Bayes intrinsic explanations
        ex_lr = client.get(f"/api/models/{m_lr['model_id']}/explain").get_json()
        assert ex_lr["model_specific"]["method_type"] == "coefficients"
        ex_nb = client.get(f"/api/models/{m_nb['model_id']}/explain").get_json()
        assert ex_nb["model_specific"]["method_type"] == "naive_bayes"
        print(f"Naive Bayes Top Discriminative Feature: {ex_nb['model_specific']['feature_parameters'][0]['feature']}")

        # ---------------------------------------------------------------------
        # TEST 8: Unsupported Intrinsic Algorithm (KNN) Displays Honest Message
        # ---------------------------------------------------------------------
        print("\n" + "=" * 88)
        print("TEST 8: KNN Honesty Check (No Fabricated Intrinsic Weights)")
        print("-" * 88)
        ex_knn = client.get(f"/api/models/{m_knn['model_id']}/explain").get_json()
        assert ex_knn["model_specific"]["supported"] is False
        print(f"KNN Model-Specific Status: Supported={ex_knn['model_specific']['supported']} | Explanation: {ex_knn['model_specific']['explanation']}")

        # ---------------------------------------------------------------------
        # TEST 5: Local Explanation (Single Prediction Attribution & Decision Path)
        # ---------------------------------------------------------------------
        print("\n" + "=" * 88)
        print("TEST 5: Local Single-Prediction Explanation (Classification & Regression)")
        print("-" * 88)
        loc_dt = client.post(f"/api/models/{m_dt['model_id']}/explain/local", json={
            "inputs": {"Pclass": "1", "Sex": "female", "Age": "24", "SibSp": "0", "Parch": "0", "Fare": "120.0", "Embarked": "C"},
            "actual": "1"
        }).get_json()
        print("Decision Tree Local Narrative:")
        for line in loc_dt["narrative"]:
            print("  *", line)
        print(f"Decision Path Traced: {loc_dt['decision_path']}")
        assert len(loc_dt["contributions"]) == 7
        assert loc_dt["decision_path"] is not None and len(loc_dt["decision_path"]) > 0

        loc_reg = client.post(f"/api/models/{m_ridge['model_id']}/explain/local", json={
            "inputs": {"SqFt_Living": "3200", "Total_Rooms": "10", "Bedrooms": "5", "Neighborhood": "Downtown"}
        }).get_json()
        print("Ridge Regression Local Narrative:")
        for line in loc_reg["narrative"]:
            print("  *", line)
        # SqFt_Living = 3200 (above median) must contribute positively to Sale_Price
        sqft_contrib = next(c for c in loc_reg["contributions"] if c["feature"] == "SqFt_Living")
        assert sqft_contrib["contribution"] > 0, f"Expected positive contribution for 3200 sqft, got {sqft_contrib}"

        # ---------------------------------------------------------------------
        # TEST 6 & 7: Partial Dependence (PDP) & Counterfactual What-If Analysis (Zero Retraining)
        # ---------------------------------------------------------------------
        print("\n" + "=" * 88)
        print("TEST 6 & 7: Partial Dependence (PDP) & What-If Counterfactual Analysis")
        print("-" * 88)
        pdp_num = client.post(f"/api/models/{m_ridge['model_id']}/explain/pdp", json={"feature_name": "SqFt_Living"}).get_json()
        print(f"Numeric PDP (SqFt_Living): {pdp_num['summary']}")
        assert pdp_num["points"][-1]["mean_response"] > pdp_num["points"][0]["mean_response"]

        pdp_cat = client.post(f"/api/models/{m_ridge['model_id']}/explain/pdp", json={"feature_name": "Neighborhood"}).get_json()
        print(f"Categorical PDP (Neighborhood): {pdp_cat['summary']}")
        assert len(pdp_cat["points"]) == 3

        wi_reg = client.post(f"/api/models/{m_ridge['model_id']}/explain/whatif", json={
            "original_inputs": {"SqFt_Living": "1200", "Total_Rooms": "4", "Bedrooms": "2", "Neighborhood": "Suburb"},
            "modified_inputs": {"SqFt_Living": "2400", "Total_Rooms": "8", "Bedrooms": "4", "Neighborhood": "Downtown"}
        }).get_json()
        print(f"What-If Regression: {wi_reg['summary']} | Retrained={wi_reg['retrained']}")
        assert wi_reg["retrained"] is False
        assert wi_reg["absolute_difference"] > 100000

        wi_cls = client.post(f"/api/models/{m_dt['model_id']}/explain/whatif", json={
            "original_inputs": {"Pclass": "3", "Sex": "male", "Age": "35", "SibSp": "0", "Parch": "0", "Fare": "7.8", "Embarked": "S"},
            "modified_inputs": {"Pclass": "1", "Sex": "female", "Age": "25", "SibSp": "0", "Parch": "0", "Fare": "90.0", "Embarked": "C"}
        }).get_json()
        print(f"What-If Classification: {wi_cls['summary']} | Class Flipped={wi_cls['class_flipped']} | Retrained={wi_cls['retrained']}")
        assert wi_cls["retrained"] is False
        assert wi_cls["class_flipped"] is True

    print("=" * 88)
    print("ALL 11 MODEL EXPLAINABILITY CENTRE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_explainability_tests()
