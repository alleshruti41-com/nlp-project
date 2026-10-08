"""
Itinerary Generation Module for Smart Budget Trip Planner
==========================================================
Generates personalized, day-wise itineraries dynamically based on:
- Destination name and characteristics
- Trip duration (number of days)
- User preferences (e.g., Adventure, Beach, Food, Culture)
- Popular Activities and Keywords from the dataset
"""

from typing import Dict, List, Any, Optional
import random

# Regional food recommendations based on destination or state
LOCAL_FOOD_MAP = {
    "Goa": "Goan Fish Curry, Prawn Balchao, Bebinca & Feni",
    "Manali": "Siddu, Trout fish, Thukpa & steaming Momos",
    "Jaipur": "Dal Baati Churma, Pyaaz Kachori & Ghevar",
    "Udaipur": "Gatte ki Sabzi, Laal Maas & Ker Sangri",
    "Kerala": "Appam with Stew, Kerala Sadya & Karimeen Pollichathu",
    "Mumbai": "Vada Pav, Pav Bhaji, Bombay Duck & Chowpatty Kulfi",
    "Delhi": "Chole Bhature, Butter Chicken, Paranthas & Chandni Chowk Chaat",
    "Agra": "Petha (Agra Special), Bedmi Puri & Mughlai Kebabs",
    "Rishikesh": "Garhwali Kafuli, Ayurvedic Herbal Tea & Riverside Cafe Bowls",
    "Ooty": "Nilgiri Varkey, Fresh Homemade Chocolates & Ooty Tea",
    "Mysore": "Mysore Pak, Mysore Masala Dosa & Filter Coffee",
    "Darjeeling": "Darjeeling First Flush Tea, Tibetan Momos, Thukpa & Churpee",
    "Shimla": "Chha Gosht, Madra & Himachali Dhaam",
    "Pune": "Misal Pav, Bakarwadi & SPDP (Sev Potato Dahi Puri)",
    "Lonavala": "Maganlal Chikki, Fresh Fudge & Corn Pakodas",
    "Mahabaleshwar": "Fresh Strawberries with Cream & Corn Patties",
    "Amritsar": "Amritsari Kulcha with Chole, Makki Roti & Lassi",
    "Varanasi": "Kachori Jalebi, Banarasi Paan, Malaiyo & Tamatar Chaat",
    "Hyderabad": "Dum Biryani, Haleem, Mirchi ka Salan & Double ka Meetha",
    "Bengaluru": "Benne Dosa, Bisi Bele Bath & Filter Coffee",
    "Gokarna": "Fresh Fish Thali, Gadbad Ice Cream & Nutella Pancakes",
    "Andaman": "Fresh Crab Masala, Lobster, Coconut Prawn Curry & Fish Tikka",
    "Jaisalmer": "Ker Sangri, Bajra Roti with Ghee, & Mawa Kachori",
    "Coorg": "Pandi Curry (Coorg Pork/Veg curry), Akki Roti & Fresh Estate Coffee",
    "Nashik": "Misal Pav, Grape Wine Tasting & Sula Artisan Cheese"
}

# Local transit suggestions
LOCAL_TRANSIT_MAP = {
    "Goa": "Rented scooter / self-drive car (~₹400/day)",
    "Manali": "Shared taxis or local buses for valley points",
    "Jaipur": "Auto-rickshaws, E-rickshaws and Jaipur Metro",
    "Udaipur": "Auto-rickshaw or scenic lakeside walking",
    "Kerala": "KSRTC buses, state passenger ferries & auto-rickshaws",
    "Mumbai": "Local trains, Mumbai Metro, and black-and-yellow Kaali Peeli taxis",
    "Delhi": "Delhi Metro (fastest & most economical) and E-rickshaws",
    "Agra": "Battery-operated auto-rickshaws and tongas",
    "Rishikesh": "Walking across suspension bridges & shared tempos",
    "Ooty": "Local hill buses, shared cabs & historic Toy Train",
    "Mysore": "KSRTC city buses & tongas around palace area",
    "Darjeeling": "Shared Sumo cabs and scenic walking trails",
    "Shimla": "Walking on Mall Road (pedestrian-only) & HRTC buses",
    "Pune": "PMPML city buses, auto-rickshaws & Metro",
    "Lonavala": "Local auto-rickshaws or shared private vans",
    "Mahabaleshwar": "Union tourist taxis or personal vehicle",
    "Amritsar": "E-rickshaws and Golden Temple free shuttle buses",
    "Varanasi": "Cycle-rickshaws, E-rickshaws & hand-rowed wooden boats",
    "Hyderabad": "Hyderabad Metro, RTC buses and auto-rickshaws",
    "Bengaluru": "Namma Metro and BMTC bench buses",
    "Gokarna": "Auto-rickshaws and hiking coastal trails on foot",
    "Andaman": "Government passenger ferries & rented scooters",
    "Jaisalmer": "Camel carts, auto-rickshaws and walking within the living fort",
    "Coorg": "Private rental cabs or local private buses",
    "Nashik": "City buses, shared autos, and wine-tour cabs"
}


def parse_destination_activities(raw_activities: str) -> List[str]:
    """
    Parses semicolon or comma separated activities into a clean list.
    """
    if not raw_activities or not isinstance(raw_activities, str):
        return []
    delimiter = ";" if ";" in raw_activities else ","
    acts = [a.strip() for a in raw_activities.split(delimiter) if a.strip()]
    return acts


def generate_day_wise_itinerary(
    destination: str,
    days: int,
    user_preferences: Optional[List[str]] = None,
    destination_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Step 7: Dynamically builds day-by-day itinerary tailored to
    destination specifics, user preferences, and available activities.
    """
    days = max(1, int(days))
    user_preferences = user_preferences or []
    destination_data = destination_data or {}

    raw_acts = destination_data.get("Popular_Activities", "")
    activities = parse_destination_activities(raw_acts)
    dest_type = destination_data.get("Type", "General")
    state = destination_data.get("State", "India")
    keywords = destination_data.get("Keywords", "")

    # Default fallback activities if empty
    if not activities:
        activities = [
            f"Historic city tour and major monuments in {destination}",
            f"Scenic nature exploration and viewpoints around {destination}",
            f"Vibrant local markets and souvenir shopping in {destination}",
            f"Cultural immersion and local culinary trail in {destination}",
            f"Relaxing sunset visit and photography at famous landmarks"
        ]

    # Culinary and Transit pointers
    local_food = LOCAL_FOOD_MAP.get(destination, f"Famous regional cuisine and street food of {state}")
    transit_mode = LOCAL_TRANSIT_MAP.get(destination, "Local buses, auto-rickshaws or shared cabs")

    itinerary_days = []
    act_count = len(activities)

    for day_num in range(1, days + 1):
        # Pick relevant activity for the day
        act_primary = activities[(day_num - 1) % act_count]
        act_secondary = activities[(day_num) % act_count] if act_count > 1 else "Local leisure walk"

        if day_num == 1:
            # First Day: Arrival & Orientation
            title = f"Day 1: Arrival & Welcome to {destination}"
            morning = f"Arrive at {destination} (railway station/airport). Transit to your accommodation, check-in, unpack and freshen up."
            afternoon = f"Enjoy a relaxed lunch savoring local flavors. Begin orientation with a gentle stroll around the local neighborhood and nearby landmarks."
            evening = f"Head to a scenic local viewpoint or promenade for sunset. Immerse in the vibrant evening ambiance."
            night = f"Kick off your trip with a delicious welcome dinner: try {local_food.split(',')[0]} at a popular local eatery."
            focus = "Arrival, Hotel Check-In & Local Orientation"

        elif day_num == days:
            # Last Day: Wrap-up & Departure
            title = f"Day {day_num}: Final Sightseeing, Souvenirs & Farewell"
            morning = f"Savor a hearty breakfast. Take a morning heritage walk or capture final photographs around iconic spots."
            afternoon = f"Visit local markets for souvenir shopping (spices, handicrafts, and specialty sweets). Pack up and check-out from hotel."
            evening = f"Enjoy a relaxing coffee or light meal. Transfer to the transit hub (station/airport) for your return journey."
            night = f"Board return journey with unforgettable memories of {destination}!"
            focus = "Sightseeing, Shopping, Checkout & Departure"

        elif day_num == 2:
            # Day 2: Prime attraction / signature experience
            title = f"Day 2: Signature Exploration - {act_primary}"
            morning = f"Start early with energy! Head out for the signature experience: {act_primary}."
            afternoon = f"Lunch break at an authentic restaurant. Continue with relaxed sightseeing: {act_secondary}."
            evening = f"Explore popular bustling streets, cafes, or waterfront spots. Enjoy sunset photography."
            night = f"Dinner featuring regional delicacies ({local_food}). Stroll through local night markets."
            focus = f"Highlight Activity: {act_primary}"

        else:
            # Intermediate Days (Days 3..N-1): Preference-tuned exploration
            title = f"Day {day_num}: In-Depth Adventure & Culture"
            morning = f"Morning excursion: Experience {act_primary}. Soak in the picturesque natural and architectural sights."
            afternoon = f"Midday authentic culinary stop. Visit artisanal workshops or hidden gems recommended by locals: {act_secondary}."
            evening = f"Spend the golden hour soaking in the destination vibe, relaxing at scenic spots or participating in cultural shows."
            night = f"Dinner at a charming restaurant. Wind down and plan for the next day."
            focus = f"Cultural & Sightseeing: {act_primary}"

        itinerary_days.append({
            "day_number": day_num,
            "title": title,
            "focus": focus,
            "morning": morning,
            "afternoon": afternoon,
            "evening": evening,
            "night": night,
            "recommended_food": local_food.split(",")[min(day_num - 1, len(local_food.split(",")) - 1)].strip()
        })

    return {
        "destination": destination,
        "state": state,
        "total_days": days,
        "type": dest_type,
        "local_cuisine": local_food,
        "recommended_transit": transit_mode,
        "days_itinerary": itinerary_days,
        "budget_travel_tips": [
            f"💡 **Transport Tip:** Use {transit_mode} to avoid inflated tourist fares.",
            f"🍛 **Culinary Tip:** Must try {local_food} at crowded local diners where food is freshest.",
            "⏰ **Timing Tip:** Start morning sightseeing before 9:00 AM to beat tourist queues and harsh afternoon heat.",
            "🎟️ **Ticketing Tip:** Pre-book or look for state combo tickets for monuments to get student discounts."
        ]
    }


if __name__ == "__main__":
    dest_data = {
        "Destination": "Goa",
        "State": "Goa",
        "Type": "Beach",
        "Popular_Activities": "Scuba diving; Water sports at Calangute; Beach hopping at Baga; Nightlife & beach clubs; Fort Aguada visit",
        "Keywords": "beach sea seafood adventure"
    }
    itin = generate_day_wise_itinerary("Goa", days=4, user_preferences=["Beach", "Adventure"], destination_data=dest_data)
    print("Generated Itinerary for", itin["destination"], f"({itin['total_days']} Days):")
    for d in itin["days_itinerary"]:
        print(f"\n{d['title']}")
        print("  Morning:", d["morning"])
        print("  Afternoon:", d["afternoon"])
        print("  Evening:", d["evening"])
        print("  Night:", d["night"])
