"""
Budget Calculation and Optimization Module
===========================================
Calculates itemized trip expenses dynamically according to the formula:
- Transportation: Transport_Cost × Number_of_Travelers
- Accommodation: Accommodation_Per_Day × Number_of_Days × Number_of_Travelers
- Food: Food_Per_Day × Number_of_Days × Number_of_Travelers
- Activities: Activity_Cost_Per_Day × Number_of_Days × Number_of_Travelers
- Other: Other_Cost_Per_Day × Number_of_Days × Number_of_Travelers

Total Trip Cost = Transportation + Accommodation + Food + Activities + Other

Also provides dynamic Budget Optimization when a trip exceeds the user's budget.
"""

from typing import Dict, Any, Union
import pandas as pd


def estimate_destination_cost(row: Union[pd.Series, dict], travelers: int, days: int) -> float:
    """
    Computes total estimated trip cost for a destination row.
    """
    transport = float(row.get("Transport_Cost", 0)) * travelers
    acc = float(row.get("Accommodation_Per_Day", 0)) * days * travelers
    food = float(row.get("Food_Per_Day", 0)) * days * travelers
    act = float(row.get("Activity_Cost_Per_Day", 0)) * days * travelers
    other = float(row.get("Other_Cost_Per_Day", 0)) * days * travelers
    return transport + acc + food + act + other


def calculate_trip_budget(
    destination_data: Union[pd.Series, dict],
    travelers: int,
    days: int,
    available_budget: float
) -> Dict[str, Any]:
    """
    Step 5: Calculates dynamic expense breakdown and compares against available budget.
    """
    travelers = max(1, int(travelers))
    days = max(1, int(days))
    available_budget = float(available_budget)

    # Unit values from destination data
    unit_transport = float(destination_data.get("Transport_Cost", 0))
    unit_acc = float(destination_data.get("Accommodation_Per_Day", 0))
    unit_food = float(destination_data.get("Food_Per_Day", 0))
    unit_act = float(destination_data.get("Activity_Cost_Per_Day", 0))
    unit_other = float(destination_data.get("Other_Cost_Per_Day", 0))

    # Exact formulas as specified
    transport_cost = unit_transport * travelers
    accommodation_cost = unit_acc * days * travelers
    food_cost = unit_food * days * travelers
    activity_cost = unit_act * days * travelers
    other_cost = unit_other * days * travelers

    total_trip_cost = (
        transport_cost
        + accommodation_cost
        + food_cost
        + activity_cost
        + other_cost
    )

    is_within_budget = total_trip_cost <= available_budget

    if is_within_budget:
        remaining_budget = available_budget - total_trip_cost
        exceeded_amount = 0.0
        status_message = "✅ TRIP IS WITHIN YOUR BUDGET"
    else:
        remaining_budget = 0.0
        exceeded_amount = total_trip_cost - available_budget
        status_message = "⚠️ TRIP EXCEEDS YOUR BUDGET"

    return {
        "travelers": travelers,
        "days": days,
        "available_budget": available_budget,
        "transport_cost": transport_cost,
        "accommodation_cost": accommodation_cost,
        "food_cost": food_cost,
        "activity_cost": activity_cost,
        "other_cost": other_cost,
        "total_trip_cost": total_trip_cost,
        "per_traveler_cost": total_trip_cost / travelers if travelers > 0 else 0,
        "is_within_budget": is_within_budget,
        "remaining_budget": remaining_budget,
        "exceeded_amount": exceeded_amount,
        "status_message": status_message
    }


def optimize_trip_budget(
    budget_breakdown: Dict[str, Any],
    available_budget: float
) -> Dict[str, Any]:
    """
    Step 6: If the trip exceeds the user's budget, automatically attempts
    to optimize expenses through realistic, actionable cost reductions.
    """
    original_total = budget_breakdown["total_trip_cost"]
    
    if original_total <= available_budget:
        return {
            "needed_optimization": False,
            "original_cost": original_total,
            "optimized_cost": original_total,
            "amount_saved": 0.0,
            "fits_within_budget": True,
            "remaining_budget": available_budget - original_total,
            "exceeded_amount": 0.0,
            "status_message": "✅ TRIP IS WITHIN YOUR BUDGET",
            "optimized_breakdown": {
                "transport_cost": budget_breakdown["transport_cost"],
                "accommodation_cost": budget_breakdown["accommodation_cost"],
                "food_cost": budget_breakdown["food_cost"],
                "activity_cost": budget_breakdown["activity_cost"],
                "other_cost": budget_breakdown["other_cost"]
            },
            "suggestions": []
        }

    # Optimization strategies & savings factors
    # 1. Switch to budget hostels/guesthouses / twin sharing: ~35% off accommodation
    opt_accommodation = round(budget_breakdown["accommodation_cost"] * 0.65, 2)
    # 2. Free walking tours, nature exploration, group activity discounts: ~40% off activities
    opt_activity = round(budget_breakdown["activity_cost"] * 0.60, 2)
    # 3. Cut miscellaneous discretionary spending: ~40% off other
    opt_other = round(budget_breakdown["other_cost"] * 0.60, 2)
    # 4. Local authentic eateries & street food over hotels: ~20% off food
    opt_food = round(budget_breakdown["food_cost"] * 0.80, 2)
    # 5. Advance booking / sleeper train / state transport: ~15% off transport
    opt_transport = round(budget_breakdown["transport_cost"] * 0.85, 2)

    optimized_total = round(opt_transport + opt_accommodation + opt_food + opt_activity + opt_other, 2)
    amount_saved = round(original_total - optimized_total, 2)

    fits_now = optimized_total <= available_budget

    if fits_now:
        rem = round(available_budget - optimized_total, 2)
        exc = 0.0
        msg = "✅ Optimized trip fits within your budget."
    else:
        rem = 0.0
        exc = round(optimized_total - available_budget, 2)
        msg = f"⚠️ Even after optimization, trip exceeds budget by ₹{exc:,.2f}."

    suggestions = [
        f"🏨 **Accommodation:** Switch to certified budget hostels, guesthouses, or homestays (Saved ₹{budget_breakdown['accommodation_cost'] - opt_accommodation:,.0f}).",
        f"🎯 **Sightseeing & Activities:** Explore free viewpoints, self-guided trails, and off-peak activity passes (Saved ₹{budget_breakdown['activity_cost'] - opt_activity:,.0f}).",
        f"🍛 **Dining:** Savor authentic local eateries and popular food streets instead of resort dining (Saved ₹{budget_breakdown['food_cost'] - opt_food:,.0f}).",
        f"🚆 **Transport:** Opt for advance train reservations, shared cabs, or state buses (Saved ₹{budget_breakdown['transport_cost'] - opt_transport:,.0f}).",
        f"🛍️ **Miscellaneous:** Minimize impulse retail and negotiate local transit fares (Saved ₹{budget_breakdown['other_cost'] - opt_other:,.0f})."
    ]

    return {
        "needed_optimization": True,
        "original_cost": original_total,
        "optimized_cost": optimized_total,
        "amount_saved": amount_saved,
        "fits_within_budget": fits_now,
        "remaining_budget": rem,
        "exceeded_amount": exc,
        "status_message": msg,
        "optimized_breakdown": {
            "transport_cost": opt_transport,
            "accommodation_cost": opt_accommodation,
            "food_cost": opt_food,
            "activity_cost": opt_activity,
            "other_cost": opt_other
        },
        "suggestions": suggestions
    }


if __name__ == "__main__":
    dummy_dest = {
        "Destination": "Goa",
        "Transport_Cost": 2500,
        "Accommodation_Per_Day": 1200,
        "Food_Per_Day": 800,
        "Activity_Cost_Per_Day": 600,
        "Other_Cost_Per_Day": 300
    }
    b = calculate_trip_budget(dummy_dest, travelers=2, days=4, available_budget=20000)
    print("Initial Calculation:", b)
    opt = optimize_trip_budget(b, available_budget=20000)
    print("Optimization Result:", opt["status_message"], "Saved:", opt["amount_saved"])
