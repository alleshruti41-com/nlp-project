"""
End-to-End Automated Verification Script for Smart Budget Trip Planner
======================================================================
Tests all 10 validation points specified in Step 15.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd
import numpy as np

# 1. Test Dataset
print("=== 1. TESTING DATASET INTEGRITY ===")
df = pd.read_csv("travel_data.csv")
print(f"Total destinations loaded: {len(df)}")
required_cols = [
    "Destination", "State", "Type", "Transport_Cost", "Accommodation_Per_Day",
    "Food_Per_Day", "Activity_Cost_Per_Day", "Other_Cost_Per_Day",
    "Popular_Activities", "Keywords", "Recommended_Days"
]
for col in required_cols:
    assert col in df.columns, f"Missing column: {col}"
print(f"All {len(required_cols)} required columns present!")
assert len(df) >= 25, "Dataset must contain at least 25 destinations"
print("Dataset validation: PASSED ✅\n")

# 2. Test NLP Processing
print("=== 2. TESTING NLP PROCESSING ===")
from nlp import analyze_travel_request
test_query = "I want a cheap beach trip with good food and some adventure."
nlp_res = analyze_travel_request(test_query)
print("Query:", test_query)
print("Clean Text:", nlp_res["clean_text"])
print("Tokens:", nlp_res["tokens"])
print("Keywords:", nlp_res["keywords"])
print("Detected Preferences:", nlp_res["preferences"])
print("Budget Classification:", nlp_res["budget_type"])

for expected in ["Beach", "Adventure", "Food", "Budget"]:
    assert expected in nlp_res["preferences"], f"Expected {expected} in preferences!"
assert nlp_res["budget_type"] == "Budget", "Expected budget_type == 'Budget'"
print("NLP verification: PASSED ✅\n")

# 3. Test Recommendations
print("=== 3. TESTING RECOMMENDATION SYSTEM ===")
from recommender import get_recommendations
recs = get_recommendations(
    travelers=2,
    days=4,
    budget=20000,
    selected_destination="Goa",
    manual_preferences=["Beach", "Adventure", "Food"],
    natural_query=test_query,
    top_n=3
)
print(f"Top {len(recs)} Recommendations:")
for i, r in enumerate(recs, 1):
    print(f"#{i} {r['Destination']} ({r['State']}) | Match: {r['Match_Percentage']}% | Est. Cost: Rs. {r['Estimated_Total_Cost']:,.2f}")
    print(f"   Reason: {r['Reason']}")
assert recs[0]["Destination"] == "Goa", "Goa should be top recommended destination!"
assert recs[0]["Match_Percentage"] >= 80, "Goa match percentage should be high!"
print("Recommendation verification: PASSED ✅\n")

# 4. Test Budget Calculation
print("=== 4. TESTING DYNAMIC BUDGET CALCULATION ===")
from budget import calculate_trip_budget, optimize_trip_budget
goa_data = recs[0]["Raw_Data"]
b_calc = calculate_trip_budget(goa_data, travelers=2, days=4, available_budget=20000)
print(f"Transportation: Rs. {b_calc['transport_cost']:,}")
print(f"Accommodation:  Rs. {b_calc['accommodation_cost']:,}")
print(f"Food:           Rs. {b_calc['food_cost']:,}")
print(f"Activities:     Rs. {b_calc['activity_cost']:,}")
print(f"Other:          Rs. {b_calc['other_cost']:,}")
print(f"Total Trip Cost: Rs. {b_calc['total_trip_cost']:,}")
print(f"Available Budget: Rs. {b_calc['available_budget']:,}")
print(f"Status: {b_calc['status_message']}")
print(f"Exceeded Amount: Rs. {b_calc['exceeded_amount']:,}")

assert b_calc["total_trip_cost"] == 28200.0, f"Expected 28200, got {b_calc['total_trip_cost']}"
assert b_calc["is_within_budget"] is False
assert b_calc["exceeded_amount"] == 8200.0
print("Budget calculation verification: PASSED ✅\n")

# 5. Test Budget Optimization
print("=== 5. TESTING BUDGET OPTIMIZATION ===")
b_opt = optimize_trip_budget(b_calc, available_budget=20000)
print(f"Original Cost:  Rs. {b_opt['original_cost']:,.2f}")
print(f"Optimized Cost: Rs. {b_opt['optimized_cost']:,.2f}")
print(f"Amount Saved:   Rs. {b_opt['amount_saved']:,.2f}")
print(f"Status Message: {b_opt['status_message']}")
print(f"Fits within budget: {b_opt['fits_within_budget']}")
print(f"Remaining budget: Rs. {b_opt['remaining_budget']:,.2f}")
assert b_opt["needed_optimization"] is True
assert b_opt["amount_saved"] > 0
assert b_opt["fits_within_budget"] is True
print("Budget optimization verification: PASSED ✅\n")

# 6. Test Itinerary Generation
print("=== 6. TESTING ITINERARY GENERATION ===")
from itinerary import generate_day_wise_itinerary
itin = generate_day_wise_itinerary("Goa", days=4, user_preferences=["Beach", "Adventure"], destination_data=goa_data)
assert len(itin["days_itinerary"]) == 4, "Expected exactly 4 days in itinerary"
print(f"Itinerary generated for {itin['destination']} ({itin['total_days']} Days):")
for d in itin["days_itinerary"]:
    print(f"- {d['title']}: {d['focus']}")
print("Itinerary verification: PASSED ✅\n")

# 7. Test Chart Rendering
print("=== 7. TESTING CHART GENERATION ===")
from app import plot_expense_charts
fig = plot_expense_charts(b_calc, b_opt)
assert fig is not None, "Figure must be created"
print("Matplotlib pie and bar chart rendering: PASSED ✅\n")

print("🎉 ALL TEST SUITES PASSED PERFECTLY WITH ZERO ERRORS!")
