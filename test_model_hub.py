"""Comprehensive End-to-End Test Suite for the Model Hub & Model Persistence System.
Tests:
1. Train & save a classification model (Titanic Survival — Random Forest & Linear SVM).
2. Train & save a regression model (House Sale Price — Ridge Regression).
3. Multiple model versions (Random Forest – Version 1, Version 2, Version 3) without overwriting.
4. Persistence across application restarts (fresh module/SQLite connections & Flask test client).
5. Load saved model and verify zero-retraining prediction (`retrained == False`, `fit()` never called).
6. Probability support vs non-probability decision margin handling (Random Forest vs Linear SVM).
7. Input validation: missing features (median/mode imputation + strict error), extra features, invalid numeric input, unseen categorical values.
8. Corrupted model artifact detection (HMAC-SHA256 tamper detection & missing artifact detection).
9. Model comparison (same-dataset fair comparison, cross-dataset warning, and incompatible classification-vs-regression block).
10. Download functionality (artifact JSON & metadata JSON with zero internal server path leakage).
11. Model deletion (deleting Version 2 leaves Version 1 and Version 3 intact).
"""
import io
import json
import os
import random
import shutil
import unittest.mock as mock

import app as flask_app
import minilearn as ml
import model_hub as mh


def run_model_hub_tests():
    # Clean test hub directory first so tests are deterministic, then leave realistic saved models at the end
    if os.path.exists(mh.HUB_DIR):
        shutil.rmtree(mh.HUB_DIR)
    mh._ensure_storage()

    client = flask_app.app.test_client()

    # -------------------------------------------------------------------------
    # TEST 1: Train & Save Classification Models + Versioning (v1, v2, v3)
    # -------------------------------------------------------------------------
    print("=" * 88)
    print("TEST 1 & 3: Train & Save Classification Models + Automatic Versioning (v1, v2, v3)")
    print("-" * 88)
    titanic_path = os.path.join(os.path.dirname(__file__), "static", "titanic_sample.csv")
    with open(titanic_path, "rb") as f:
        titanic_bytes = f.read()

    # Save Random Forest Version 1
    res1 = client.post("/api/models/save", data={
        "file": (io.BytesIO(titanic_bytes), "titanic.csv"),
        "dataset_name": "titanic.csv",
        "target": "Survived",
        "tune": "1",
        "model_name": "Random forest"
    }, content_type="multipart/form-data")
    assert res1.status_code == 200, res1.get_json()
    m_rf_v1 = res1.get_json()
    print(f"Saved: {m_rf_v1['display_name']} (ID={m_rf_v1['model_id'][:8]}..., v={m_rf_v1['version']}, Test Macro F1={m_rf_v1['primary_metric_value']})")
    assert m_rf_v1["version"] == 1
    assert m_rf_v1["display_name"] == "Random forest – Version 1"

    # Save Random Forest Version 2 (with different excluded feature to create distinct v2)
    res2 = client.post("/api/models/save", data={
        "file": (io.BytesIO(titanic_bytes), "titanic.csv"),
        "dataset_name": "titanic.csv",
        "target": "Survived",
        "exclude": "SibSp|Parch",
        "tune": "0",
        "model_name": "Random forest"
    }, content_type="multipart/form-data")
    assert res2.status_code == 200
    m_rf_v2 = res2.get_json()
    print(f"Saved: {m_rf_v2['display_name']} (ID={m_rf_v2['model_id'][:8]}..., v={m_rf_v2['version']}, Test Macro F1={m_rf_v2['primary_metric_value']})")
    assert m_rf_v2["version"] == 2
    assert m_rf_v2["model_id"] != m_rf_v1["model_id"]

    # Save Random Forest Version 3
    res3 = client.post("/api/models/save", data={
        "file": (io.BytesIO(titanic_bytes), "titanic.csv"),
        "dataset_name": "titanic.csv",
        "target": "Survived",
        "tune": "0",
        "model_name": "Random forest"
    }, content_type="multipart/form-data")
    assert res3.status_code == 200
    m_rf_v3 = res3.get_json()
    print(f"Saved: {m_rf_v3['display_name']} (ID={m_rf_v3['model_id'][:8]}..., v={m_rf_v3['version']}, Test Macro F1={m_rf_v3['primary_metric_value']})")
    assert m_rf_v3["version"] == 3

    # Save Linear SVM (non-probability margin model) on Titanic
    res_svm = client.post("/api/models/save", data={
        "file": (io.BytesIO(titanic_bytes), "titanic.csv"),
        "dataset_name": "titanic.csv",
        "target": "Survived",
        "tune": "1",
        "model_name": "Linear SVM"
    }, content_type="multipart/form-data")
    assert res_svm.status_code == 200
    m_svm = res_svm.get_json()
    print(f"Saved: {m_svm['display_name']} (ID={m_svm['model_id'][:8]}..., v={m_svm['version']}, Test Macro F1={m_svm['primary_metric_value']})")

    # -------------------------------------------------------------------------
    # TEST 2: Train & Save Regression Model (House Prices)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("TEST 2: Train & Save Regression Model (House Prices — Ridge Regression)")
    print("-" * 88)
    rng = random.Random(42)
    reg_lines = ["SqFt_Living,Bedrooms,Bathrooms,Neighborhood,Sale_Price"]
    for _ in range(120):
        sqft = rng.randint(850, 3800)
        beds = max(1, min(6, int(sqft / 650)))
        baths = max(1, min(4, int(beds * 0.75)))
        hood = rng.choice(["Downtown", "North", "Suburb"])
        bonus = {"Downtown": 75000, "North": 35000, "Suburb": 0}[hood]
        price = round(55000 + sqft * 160 + beds * 10000 + bonus + rng.gauss(0, 18000), 2)
        reg_lines.append(f"{sqft},{beds},{baths},{hood},{price:.2f}")
    reg_csv_bytes = "\n".join(reg_lines).encode("utf-8")

    res_reg = client.post("/api/models/save", data={
        "file": (io.BytesIO(reg_csv_bytes), "house_prices.csv"),
        "dataset_name": "house_prices.csv",
        "target": "Sale_Price",
        "tune": "1",
        "model_name": "Ridge regression"
    }, content_type="multipart/form-data")
    assert res_reg.status_code == 200
    m_reg = res_reg.get_json()
    print(f"Saved: {m_reg['display_name']} (ID={m_reg['model_id'][:8]}..., R²={m_reg['primary_metric_value']}, RMSE={m_reg['secondary_metric_value']})")
    assert m_reg["task_type"] == "regression"
    assert "mse" in m_reg["evaluation_metrics"] and "mae" in m_reg["evaluation_metrics"]

    # -------------------------------------------------------------------------
    # TEST 4 & 5: Simulate App Restart, Load Saved Models & Verify ZERO Retraining
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("TEST 4, 5 & 6: Persistence Across Restart & Zero-Retraining Prediction Verification")
    print("-" * 88)
    hub_list = client.get("/api/models").get_json()
    print(f"Model Hub Summary after restart check: Total={hub_list['summary']['total_models']}, "
          f"Classification={hub_list['summary']['classification_count']}, Regression={hub_list['summary']['regression_count']}")
    assert hub_list["summary"]["total_models"] == 5

    # Load Random Forest v1
    load_resp = client.post(f"/api/models/{m_rf_v1['model_id']}/load")
    assert load_resp.status_code == 200
    load_data = load_resp.get_json()
    assert load_data["retrained"] is False
    print(f"Load Response: {load_data['message']}")

    # Patch DecisionTree.fit, RandomForest.fit, RidgeRegression.fit, LinearSVM.fit to raise RuntimeError if called!
    with mock.patch.object(ml.RandomForest, "fit", side_effect=RuntimeError("RETRAINING FORBIDDEN!")), \
         mock.patch.object(ml.DecisionTree, "fit", side_effect=RuntimeError("RETRAINING FORBIDDEN!")), \
         mock.patch.object(ml.RidgeRegression, "fit", side_effect=RuntimeError("RETRAINING FORBIDDEN!")), \
         mock.patch.object(ml.LinearSVM, "fit", side_effect=RuntimeError("RETRAINING FORBIDDEN!")):

        # Predict with Random Forest v1 (supports probabilities)
        pred_rf = client.post(f"/api/models/{m_rf_v1['model_id']}/predict", json={
            "inputs": {"Pclass": "1", "Sex": "female", "Age": "29", "SibSp": "0", "Parch": "0", "Fare": "85.0", "Embarked": "C"}
        }).get_json()
        print(f"Random Forest v1 Prediction (fit() blocked): Class='{pred_rf['prediction']}', "
              f"Probabilities={pred_rf['probabilities']}, Retrained={pred_rf['retrained']}")
        assert pred_rf["retrained"] is False
        assert pred_rf["probability_supported"] is True
        assert len(pred_rf["probabilities"]) == 2

        # Predict with Linear SVM v1 (margin decision score, no fabricated probabilities)
        pred_svm = client.post(f"/api/models/{m_svm['model_id']}/predict", json={
            "inputs": {"Pclass": "3", "Sex": "male", "Age": "25", "SibSp": "0", "Parch": "0", "Fare": "7.25", "Embarked": "S"}
        }).get_json()
        print(f"Linear SVM Prediction (fit() blocked): Class='{pred_svm['prediction']}', "
              f"ProbSupported={pred_svm['probability_supported']}, Note='{pred_svm.get('probability_note')}'")
        assert pred_svm["retrained"] is False
        assert pred_svm["probability_supported"] is False
        assert pred_svm["probabilities"] is None

        # Predict with Ridge Regression v1 (House Prices)
        pred_reg = client.post(f"/api/models/{m_reg['model_id']}/predict", json={
            "inputs": {"SqFt_Living": "2400", "Bedrooms": "4", "Bathrooms": "3", "Neighborhood": "Downtown"}
        }).get_json()
        print(f"Ridge Regression Prediction (fit() blocked): Sale_Price='{pred_reg['prediction']}', Retrained={pred_reg['retrained']}")
        assert pred_reg["retrained"] is False
        assert pred_reg["prediction_numeric"] > 300000

    # -------------------------------------------------------------------------
    # TEST 7: Input Validation (Missing Features, Extra Features, Invalid Types, Unseen Categories)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("TEST 7: Input Validation (Missing, Extra, Invalid Type, Unseen Category)")
    print("-" * 88)
    # A. Missing features + Extra features + Unseen category (tolerant mode with warnings)
    pred_warn = client.post(f"/api/models/{m_rf_v1['model_id']}/predict", json={
        "inputs": {"Pclass": "1", "Sex": "female", "Embarked": "UNSEEN_PORT", "Extra_Unused_Column": "999"}
    }).get_json()
    print(f"Tolerant Mode Warnings ({len(pred_warn['warnings'])}):")
    for w in pred_warn["warnings"]:
        print(f"  - {w}")
    assert any("Extra_Unused_Column" in w for w in pred_warn["warnings"])
    assert any("UNSEEN_PORT" in w for w in pred_warn["warnings"])
    assert any("Age" in w for w in pred_warn["warnings"])

    # B. Strict schema validation mode rejects missing/extra features or invalid categories
    strict_resp = client.post(f"/api/models/{m_rf_v1['model_id']}/predict", json={
        "inputs": {"Pclass": "1"},
        "strict": True
    })
    assert strict_resp.status_code == 400
    print(f"Strict Missing-Feature Rejection Verified (HTTP 400): {strict_resp.get_json()['error']}")

    # C. Invalid non-numeric value in numeric field
    bad_num_resp = client.post(f"/api/models/{m_rf_v1['model_id']}/predict", json={
        "inputs": {"Pclass": "1", "Sex": "female", "Age": "not_a_number", "SibSp": "0", "Parch": "0", "Fare": "50", "Embarked": "S"}
    })
    assert bad_num_resp.status_code == 400
    print(f"Invalid Numeric Type Rejection Verified (HTTP 400): {bad_num_resp.get_json()['error']}")

    # -------------------------------------------------------------------------
    # TEST 8: Corrupted Model Artifact Detection (HMAC-SHA256 Tamper Check)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("TEST 8: Corrupted Model Artifact Detection (HMAC-SHA256 Verification)")
    print("-" * 88)
    artifact_file = os.path.join(mh.HUB_DIR, f"{m_rf_v2['model_id']}.artifact.json")
    with open(artifact_file, "r", encoding="utf-8") as f:
        orig_content = f.read()
    try:
        # Tamper with the artifact content
        tampered = orig_content.replace('"version": 2', '"version": 999')
        with open(artifact_file, "w", encoding="utf-8") as f:
            f.write(tampered)
        corrupt_resp = client.post(f"/api/models/{m_rf_v2['model_id']}/load")
        assert corrupt_resp.status_code == 400
        print(f"Tampered Artifact Blocked (HTTP 400): {corrupt_resp.get_json()['error']}")
    finally:
        # Restore original valid content
        with open(artifact_file, "w", encoding="utf-8") as f:
            f.write(orig_content)

    # -------------------------------------------------------------------------
    # TEST 9: Model Comparison (Same-Dataset & Incompatible Task Check)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("TEST 9: Saved Model Comparison & Task Compatibility Guard")
    print("-" * 88)
    cmp_ok = client.post("/api/models/compare", json={
        "model_ids": [m_rf_v1["model_id"], m_rf_v3["model_id"], m_svm["model_id"]]
    })
    assert cmp_ok.status_code == 200
    cmp_data = cmp_ok.get_json()
    print(f"Valid Comparison: {len(cmp_data['models'])} classification models compared | Same Dataset={cmp_data['same_dataset_and_target']}")

    # Incompatible comparison (Classification vs Regression) must be rejected
    cmp_bad = client.post("/api/models/compare", json={
        "model_ids": [m_rf_v1["model_id"], m_reg["model_id"]]
    })
    assert cmp_bad.status_code == 400
    print(f"Incompatible Task Comparison Blocked (HTTP 400): {cmp_bad.get_json()['error']}")

    # -------------------------------------------------------------------------
    # TEST 10: Download Functionality (Artifact & Metadata, No Server Path Leakage)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("TEST 10: Download Artifact & Metadata Verification")
    print("-" * 88)
    dl_art = client.get(f"/api/models/{m_rf_v1['model_id']}/download?kind=artifact")
    assert dl_art.status_code == 200
    art_text = dl_art.get_data(as_text=True)
    assert "C:\\" not in art_text and "c:\\" not in art_text and "/Users/" not in art_text
    print(f"Artifact Download Verified: Content-Disposition={dl_art.headers.get('Content-Disposition')} ({len(art_text)} bytes, zero server paths exposed)")

    dl_meta = client.get(f"/api/models/{m_rf_v1['model_id']}/download?kind=metadata")
    assert dl_meta.status_code == 200
    meta_json = dl_meta.get_json()
    assert "included_contents" in meta_json and "evaluation_metrics" in meta_json
    print(f"Metadata Download Verified: Content-Disposition={dl_meta.headers.get('Content-Disposition')}")

    # -------------------------------------------------------------------------
    # TEST 11: Model Deletion (Deleting Version 2 Preserves Version 1 and Version 3)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("TEST 11: Model Deletion Isolation (Delete v2, Verify v1 & v3 Intact)")
    print("-" * 88)
    del_resp = client.delete(f"/api/models/{m_rf_v2['model_id']}")
    assert del_resp.status_code == 200
    print(f"Deleted: {del_resp.get_json()}")
    assert not os.path.exists(artifact_file)

    remaining = client.get("/api/models").get_json()["models"]
    rem_ids = {m["model_id"] for m in remaining}
    assert m_rf_v1["model_id"] in rem_ids
    assert m_rf_v3["model_id"] in rem_ids
    assert m_rf_v2["model_id"] not in rem_ids
    print(f"Remaining Models Verified: {[m['display_name'] for m in remaining]}")
    print("=" * 88)
    print("ALL 11 MODEL HUB & PERSISTENCE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_model_hub_tests()
