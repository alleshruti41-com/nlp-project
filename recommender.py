"""
Recommendation System Module for Smart Budget Trip Planner
===========================================================
Dynamically computes recommendation scores for destinations based on:
1. Preference Match Score (Keyword overlap + Destination Type + TF-IDF similarity)
2. Budget Suitability Score (Ratio of estimated cost vs. available budget)
3. Duration Suitability Score (Delta between requested days and Recommended_Days)

Final Recommendation Score = Preference + Budget + Duration (Normalized 0-100%)
Returns top matching destinations with dynamic explanations.
"""

from typing import List, Dict, Any, Optional
import os
import pandas as pd
import numpy as np

from nlp import analyze_travel_request, compute_tfidf_similarity
from budget import estimate_destination_cost


def load_travel_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and validates the travel dataset CSV file.
    Handles fallbacks and missing values gracefully.
    """
    candidate_paths = [
        csv_path,
        "travel_data.csv",
        os.path.join(os.path.dirname(__file__), "travel_data.csv"),
        os.path.join(os.path.dirname(__file__), "..", "travel_data.csv"),
    ]

    df = None
    for path in candidate_paths:
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path)
                break
            except Exception:
                continue

    if df is None:
        raise FileNotFoundError("Could not locate 'travel_data.csv'. Please ensure the dataset exists.")

    # Data cleaning and type enforcement
    numeric_cols = [
        "Transport_Cost", "Accommodation_Per_Day", "Food_Per_Day",
        "Activity_Cost_Per_Day", "Other_Cost_Per_Day", "Recommended_Days"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(500)
        else:
            df[col] = 500

    string_cols = ["Destination", "State", "Type", "Popular_Activities", "Keywords"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""

    return df


def calculate_preference_score(
    row: pd.Series,
    all_prefs: List[str],
    tfidf_sim: float
) -> float:
    """
    Calculates preference match score (0 - 50 points).
    Combines explicit category overlap and TF-IDF semantic query similarity.
    """
    score = 0.0
    dest_type = str(row.get("Type", "")).lower()
    dest_keywords = str(row.get("Keywords", "")).lower()
    dest_activities = str(row.get("Popular_Activities", "")).lower()
    combined_dest_text = f"{dest_type} {dest_keywords} {dest_activities}"

    if not all_prefs:
        # Neutral baseline if no preferences given
        return 25.0 + (tfidf_sim * 25.0)

    # 1. Category / Type match (up to 20 pts)
    matched_prefs = []
    for pref in all_prefs:
        pref_lower = pref.lower()
        if pref_lower in dest_type or pref_lower in combined_dest_text:
            matched_prefs.append(pref)

    overlap_ratio = len(matched_prefs) / max(1, len(all_prefs))
    score += overlap_ratio * 30.0

    # 2. TF-IDF Semantic similarity contribution (up to 20 pts)
    score += float(tfidf_sim) * 20.0

    return min(50.0, score)


def calculate_budget_score(
    estimated_cost: float,
    available_budget: float,
    budget_type: str = "Moderate"
) -> float:
    """
    Calculates budget suitability score (0 - 35 points).
    Rewards trips that fit comfortably within budget, and acknowledges easily optimizable trips.
    """
    if available_budget <= 0:
        return 10.0

    ratio = estimated_cost / available_budget

    if budget_type == "Budget":
        if ratio <= 0.85:
            return 35.0
        elif ratio <= 1.0:
            return 33.0
        elif ratio <= 1.25:
            return 28.0
        elif ratio <= 1.45:
            return 22.0
        elif ratio <= 1.75:
            return 14.0
        else:
            return max(5.0, 35.0 - (ratio * 12.0))
    elif budget_type == "Luxury":
        if 0.8 <= ratio <= 1.3:
            return 35.0
        elif ratio < 0.8:
            return 28.0
        else:
            return max(5.0, 30.0 - (ratio * 10.0))
    else:  # Moderate / General
        if ratio <= 1.0:
            return 35.0
        elif ratio <= 1.25:
            return 29.0
        elif ratio <= 1.45:
            return 22.0
        elif ratio <= 1.75:
            return 14.0
        else:
            return max(5.0, 35.0 - (ratio * 12.0))


def calculate_duration_score(requested_days: int, recommended_days: int) -> float:
    """
    Calculates duration suitability score (0 - 15 points).
    """
    diff = abs(requested_days - recommended_days)
    if diff == 0:
        return 15.0
    elif diff == 1:
        return 12.0
    elif diff == 2:
        return 8.0
    elif diff <= 4:
        return 5.0
    else:
        return 2.0


def generate_recommendation_reason(
    destination: str,
    matched_prefs: List[str],
    estimated_cost: float,
    available_budget: float,
    budget_type: str
) -> str:
    """
    Constructs a clear, dynamic explanation string for why the destination is recommended.
    """
    reasons = []

    if matched_prefs:
        pref_str = ", ".join(matched_prefs[:3])
        reasons.append(f"Matches your {pref_str} preferences")
    else:
        reasons.append("Matches your travel profile")

    if estimated_cost <= available_budget:
        savings = available_budget - estimated_cost
        if savings > 1000:
            reasons.append(f"fits comfortably within your budget (saves ~₹{savings:,.0f})")
        else:
            reasons.append("fits within your budget")
    elif estimated_cost <= available_budget * 1.45:
        reasons.append("fits within your budget with smart budget optimization")
    else:
        reasons.append("provides top experiential value")

    reason_text = " and ".join(reasons) + "."
    return reason_text[0].upper() + reason_text[1:]


def get_recommendations(
    travelers: int,
    days: int,
    budget: float,
    selected_destination: Optional[str] = None,
    manual_preferences: Optional[List[str]] = None,
    natural_query: str = "",
    top_n: int = 5,
    csv_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Step 4: Computes dynamic recommendations for all destinations.
    Returns sorted list of top destination dicts.
    """
    df = load_travel_dataset(csv_path)

    travelers = max(1, int(travelers))
    days = max(1, int(days))
    budget = max(1.0, float(budget))
    manual_preferences = manual_preferences or []

    # 1. NLP Processing on natural language request
    nlp_results = analyze_travel_request(natural_query)
    nlp_prefs = nlp_results.get("preferences", [])
    budget_type = nlp_results.get("budget_type", "Moderate")

    # Combine manual preferences with NLP detected preferences (deduplicated)
    all_prefs = []
    for p in manual_preferences + nlp_prefs:
        if p and p not in all_prefs:
            all_prefs.append(p)

    # 2. Compute TF-IDF similarity between natural query and destination corpus
    destination_corpuses = (
        df["Type"] + " " + df["Keywords"] + " " + df["Popular_Activities"] + " " + df["Destination"]
    ).tolist()
    tfidf_sims = compute_tfidf_similarity(natural_query, destination_corpuses)

    scored_destinations = []

    for idx, row in df.iterrows():
        dest_name = row["Destination"]
        dest_state = row["State"]
        dest_type = row["Type"]
        rec_days = int(row.get("Recommended_Days", 3))

        # Dynamic trip cost estimation
        est_cost = estimate_destination_cost(row, travelers, days)

        # Matched preferences for this specific destination
        combined_text = f"{dest_type.lower()} {row['Keywords'].lower()} {row['Popular_Activities'].lower()}"
        matched_prefs = [p for p in all_prefs if p.lower() in combined_text]

        # Scoring components
        pref_score = calculate_preference_score(row, all_prefs, tfidf_sims[idx])
        budget_score = calculate_budget_score(est_cost, budget, budget_type)
        duration_score = calculate_duration_score(days, rec_days)

        # Raw total score (0 to 100)
        raw_score = pref_score + budget_score + duration_score

        # Specific user destination target bonus
        if selected_destination and selected_destination.strip().lower() == dest_name.lower():
            raw_score += 15.0

        # Normalization and realistic percentage clamping
        match_percentage = min(98, max(25, int(round(raw_score))))

        reason = generate_recommendation_reason(
            dest_name,
            matched_prefs,
            est_cost,
            budget,
            budget_type
        )

        scored_destinations.append({
            "Destination": dest_name,
            "State": dest_state,
            "Type": dest_type,
            "Estimated_Total_Cost": round(est_cost, 2),
            "Match_Percentage": match_percentage,
            "Reason": reason,
            "Popular_Activities": row.get("Popular_Activities", ""),
            "Keywords": row.get("Keywords", ""),
            "Recommended_Days": rec_days,
            "Scores_Breakdown": {
                "preference": round(pref_score, 1),
                "budget": round(budget_score, 1),
                "duration": round(duration_score, 1)
            },
            "Raw_Data": row.to_dict()
        })

    # Sort descending by match percentage, then ascending by cost
    scored_destinations.sort(key=lambda x: (x["Match_Percentage"], -x["Estimated_Total_Cost"]), reverse=True)

    return scored_destinations[:top_n]


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    recs = get_recommendations(
        travelers=2,
        days=4,
        budget=20000,
        selected_destination="Goa",
        manual_preferences=["Beach"],
        natural_query="I want a cheap beach trip with good food and some adventure.",
        top_n=3
    )
    print("\n--- TOP RECOMMENDATIONS ---")
    for r in recs:
        print(f"{r['Destination']} ({r['State']}) | Match: {r['Match_Percentage']}% | Cost: Rs. {r['Estimated_Total_Cost']:,.2f}")
        print(f"Reason: {r['Reason']}\n")
