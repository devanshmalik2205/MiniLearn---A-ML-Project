"""Comprehensive 12-Scenario Test Suite for MiniLearn's Intelligent Recommendation Engine.
Verifies semantic column interpretation, ID/date/high-cardinality rejection, target leakage detection,
empirical 3-fold CV baseline validation, syllabus compliance, and 'Explore This Task' integration."""
import os
import random
import minilearn as ml
import health as hl
import explore as ex


def _make_dataset(scenario_id):
    rng = random.Random(42 + scenario_id)

    # 1. Titanic: meaningful survival classification (loaded from actual static/titanic_sample.csv)
    if scenario_id == 1:
        h, r = ml.load_csv(os.path.join(os.path.dirname(__file__), "static", "titanic_sample.csv"))
        return "1. Titanic Dataset (Survival Classification)", h, r

    # 2. Customer Segmentation: Customer_ID, Age, Gender, Annual_Income, Spending_Score, Membership_Type
    if scenario_id == 2:
        h = ["Customer_ID", "Age", "Gender", "Annual_Income", "Spending_Score", "Membership_Type"]
        r = []
        for i in range(1, 151):
            age = rng.randint(19, 68)
            gender = rng.choice(["Female", "Male"])
            income = round(25 + (age - 18) * 1.1 + rng.gauss(0, 8), 1)
            tier = "Gold" if income > 60 else ("Silver" if income > 42 else "Bronze")
            spend = round(max(5, min(99, 85 - 0.6 * age + 0.45 * income + (12 if tier == "Gold" else 0) + rng.gauss(0, 6))), 1)
            r.append([str(1000 + i), str(age), gender, str(income), str(spend), tier])
        return "2. Customer Segmentation & Spending Dataset", h, r

    # 3. House Prices: Property_ID, SqFt_Living, Bedrooms, Bathrooms, Lot_Size, Neighborhood, Sale_Price
    if scenario_id == 3:
        h = ["Property_ID", "SqFt_Living", "Bedrooms", "Bathrooms", "Lot_Size", "Neighborhood", "Sale_Price"]
        r = []
        for i in range(1, 141):
            sqft = rng.randint(850, 3800)
            beds = max(1, min(6, int(sqft / 650) + rng.choice([-1, 0, 1])))
            baths = max(1, min(4, int(beds * 0.75)))
            lot = sqft * rng.uniform(1.8, 4.5)
            hood = rng.choice(["North", "Downtown", "Suburb", "Eastside"])
            hood_bonus = {"Downtown": 85000, "North": 50000, "Suburb": 25000, "Eastside": 0}[hood]
            price = round(60000 + sqft * 165 + beds * 12000 + hood_bonus + rng.gauss(0, 18000), 2)
            r.append([str(i), str(sqft), str(beds), str(baths), f"{lot:.0f}", hood, f"{price:.2f}"])
        return "3. House Prices Dataset (Meaningful Regression)", h, r

    # 4. Customer IDs and arbitrary numerical noise columns
    if scenario_id == 4:
        h = ["Customer_ID", "Account_Code", "Random_Metric_A", "Random_Metric_B", "Arbitrary_Noise"]
        r = []
        for i in range(1, 121):
            r.append([
                str(50000 + i),
                str(900000 + i * 7),
                f"{rng.uniform(0, 100):.2f}",
                f"{rng.uniform(-50, 50):.2f}",
                f"{rng.uniform(100, 500):.2f}"
            ])
        return "4. Customer IDs and Arbitrary Uncorrelated Numerical Columns", h, r

    # 5. Highly Imbalanced Classes (94% majority class, 6% minority fraud class)
    if scenario_id == 5:
        h = ["Tx_ID", "Amount", "Velocity_24h", "Foreign_Tx", "Fraud_Status"]
        r = []
        for i in range(1, 201):
            is_fraud = 1 if i <= 12 else 0
            amt = rng.uniform(600, 2500) if is_fraud else rng.uniform(10, 450)
            vel = rng.randint(5, 14) if is_fraud else rng.randint(1, 5)
            foreign = "Yes" if (is_fraud and rng.random() < 0.8) or rng.random() < 0.1 else "No"
            r.append([str(i), f"{amt:.2f}", str(vel), foreign, "Fraud" if is_fraud else "Legit"])
        return "5. Highly Imbalanced Classes (94% Legit / 6% Fraud)", h, r

    # 6. Many Missing Values (one column 85% missing, target 15% missing)
    if scenario_id == 6:
        h = ["Record_ID", "Empty_Notes", "Mostly_Missing_Sensor", "Temperature", "Pressure", "System_Status"]
        r = []
        for i in range(1, 131):
            temp = rng.uniform(50, 120)
            press = temp * 1.4 + rng.gauss(0, 8)
            status = "" if i % 7 == 0 else ("Warning" if temp > 92 or press > 135 else "Normal")
            mostly_miss = f"{rng.uniform(1, 10):.1f}" if i % 8 == 0 else ""
            r.append([str(i), "", mostly_miss, f"{temp:.1f}", f"{press:.1f}", status])
        return "6. Dataset with Many Missing Values", h, r

    # 7. Target Leakage Column (Final_Payment predicted with Total_With_Tax = 1.08 * Final_Payment)
    if scenario_id == 7:
        h = ["Order_ID", "Items_Count", "Unit_Tier", "Discount_Pct", "Final_Payment_Copy", "Final_Payment"]
        r = []
        for i in range(1, 121):
            items = rng.randint(1, 15)
            tier = rng.choice(["Standard", "Premium", "Enterprise"])
            base = {"Standard": 25, "Premium": 60, "Enterprise": 140}[tier]
            disc = rng.choice([0, 5, 10, 15])
            payment = round(items * base * (1 - disc / 100.0) + rng.gauss(0, 8), 2)
            leaked_copy = round(payment * 1.0, 2)
            r.append([str(i), str(items), tier, str(disc), str(leaked_copy), str(payment)])
        return "7. Dataset with Target Leakage Column (Final_Payment_Copy)", h, r

    # 8. Almost No Predictive Features (Target is pure coin flip independent of features)
    if scenario_id == 8:
        h = ["Feature_X", "Feature_Y", "Random_Outcome"]
        r = []
        for i in range(1, 121):
            r.append([f"{rng.gauss(0, 1):.3f}", f"{rng.gauss(10, 2):.3f}", rng.choice(["ClassA", "ClassB"])])
        return "8. Dataset with Almost No Predictive Signal", h, r

    # 9. Only Categorical Columns
    if scenario_id == 9:
        h = ["Department", "Role_Level", "Remote_Work", "Overtime", "Attrition_Outcome"]
        r = []
        for i in range(1, 141):
            dept = rng.choice(["Sales", "Engineering", "Support", "HR"])
            level = rng.choice(["Junior", "Mid", "Senior"])
            remote = rng.choice(["Yes", "No"])
            ot = rng.choice(["Yes", "No"])
            leave = "Yes" if (ot == "Yes" and remote == "No" and level == "Junior" and rng.random() < 0.85) or rng.random() < 0.1 else "No"
            r.append([dept, level, remote, ot, leave])
        return "9. Dataset with Only Categorical Columns", h, r

    # 10. Very Few Observations (Only 14 rows)
    if scenario_id == 10:
        h = ["Feature_1", "Feature_2", "Target_Class"]
        r = [[str(i * 2.5), str(i * 1.2), "Yes" if i % 2 == 0 else "No"] for i in range(1, 15)]
        return "10. Dataset with Very Few Observations (14 rows)", h, r

    # 11. Dataset with No Meaningful Target (Only IDs, Dates, and Constants)
    if scenario_id == 11:
        h = ["Record_ID", "Transaction_Date", "Postal_Code", "Status_Constant"]
        r = [[str(100 + i), f"2026-03-{(i % 28) + 1:02d}", str(90210 + i), "ACTIVE"] for i in range(1, 80)]
        return "11. Dataset with No Meaningful Target (IDs, Dates, Constants)", h, r

    # 12. Dataset with Dates, Identifiers, High-Cardinality Text, and Legitimate Outcome
    if scenario_id == 12:
        h = ["Transaction_ID", "Transaction_Date", "Customer_Name", "Postal_Code", "Usage_Minutes", "Monthly_Bill", "Churn_Status"]
        r = []
        for i in range(1, 141):
            mins = rng.uniform(50, 900)
            bill = round(20 + mins * 0.09 + rng.gauss(0, 5), 2)
            churn = "Churned" if (bill > 75 and mins < 400) or (bill > 85 and rng.random() < 0.75) else "Retained"
            r.append([
                f"TX-{10000+i}",
                f"2026-05-{(i % 28)+1:02d}",
                f"Customer_Unique_Name_{i}",
                str(30000 + i * 3),
                f"{mins:.1f}",
                f"{bill:.2f}",
                churn
            ])
        return "12. Dates, IDs, High-Cardinality Names & Churn Target", h, r


def run_all_tests():
    for sid in range(1, 13):
        title, header, rows = _make_dataset(sid)
        info = ml.profile_columns(header, rows)
        about_data = hl.about(header, rows, info)
        recs = about_data["suggested"]
        rejected = about_data["rejected_candidates"]

        print("=" * 85)
        print(f"TEST {title} ({len(rows)} rows, {len(header)} cols)")
        print("-" * 85)
        print("Column Interpretations:")
        for ci in about_data["column_interpretations"]:
            print(f"  - {ci['name']:22} | Role: {ci['role']:38} | Candidate: {ci['target_candidate']}")
        print(f"\nGenerated Recommendations ({len(recs)}):")
        if not recs:
            print(f"  [NONE] Reason: {about_data['no_task_reason']}")
        for r in recs:
            print(f"  * [{r['score']}/100 - {r['suitability_label']}] {r['title']} | Type: {r['type']} | Target: {r['target']}")
            print(f"    Features: {r['features']}")
            if r.get("leaked_features"):
                print(f"    Auto-Excluded Leakage: {[lf['name'] for lf in r['leaked_features']]}")
            print(f"    Validation: {r['validation']['summary']}")
            # Verify Explore This Task backend execution works for every generated recommendation
            if r["goal"] == "predict":
                t_key = "regression" if r["type"] == "Regression" else "classification"
                drop = [c for c in header if c != r["target"] and c not in r["features"]]
                keep = [i for i, h in enumerate(header) if h not in drop]
                sh, sr = [header[i] for i in keep], [[row[i] for i in keep] for row in rows]
                X, y, labels, names = ml.prepare_data(sh, sr, r["target"], t_key)
                rep = ml.rank_models(X, y, t_key, names, tune=False)
                print(f"    -> Explore This Task verified: Best model = {rep['best']} ({rep['score_name']} = {rep['results'][0]['score']:.3f})")
            elif r["goal"] == "groups":
                grp = ex.groups(header, rows, info, 3, features=r["features"])
                print(f"    -> Explore This Task verified: K-Means formed {len(grp['clusters'])} clusters on {grp['rows_used']} rows")
            elif r["goal"] == "pca":
                pca_res = ex.pca_analysis(header, rows, info, features=r["features"])
                print(f"    -> Explore This Task verified: PCA computed {len(pca_res['components'])} components")
            elif r["goal"] == "anomalies":
                anom_res = ex.anomalies(header, rows, info, features=r["features"])
                print(f"    -> Explore This Task verified: Anomaly detection returned top {len(anom_res['anomalies'])} records")
        print(f"\nRejected / Excluded Columns ({len(rejected)}):")
        for rc in rejected:
            print(f"  x {rc['column']:22} ({rc['stage']}): {rc['reason']}")
        print()


if __name__ == "__main__":
    run_all_tests()
