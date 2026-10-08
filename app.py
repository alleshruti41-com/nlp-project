"""
Smart Budget Trip Planner
=========================
Streamlit Application for BE Computer Engineering NLP Mini Project
Clean, attractive, and simple demo interface illustrating the 5-step flow:
Input → NLP Preferences → Recommendation → Budget → Itinerary
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Ensure local module imports work smoothly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from nlp import analyze_travel_request
from recommender import get_recommendations, load_travel_dataset
from budget import calculate_trip_budget, optimize_trip_budget
from itinerary import generate_day_wise_itinerary

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Smart Budget Trip Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Polished, clean travel styling
st.markdown("""
<style>
    /* Main header banner */
    .hero-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 22px 28px;
        border-radius: 10px;
        margin-bottom: 22px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .hero-header h1 {
        color: #ffffff !important;
        font-size: 2.1rem;
        margin: 0 0 4px 0;
        font-weight: 700;
    }
    .hero-header p {
        color: #e2e8f0 !important;
        font-size: 1.05rem;
        margin: 0;
    }
    
    /* Step section header */
    .step-banner {
        background-color: #f1f5f9;
        border-left: 5px solid #2563eb;
        padding: 10px 16px;
        border-radius: 6px;
        margin: 20px 0 12px 0;
        font-weight: 700;
        font-size: 1.15rem;
        color: #1e293b;
    }
    
    /* Cards and boxes */
    .dest-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    .dest-card-top {
        border-top: 4px solid #2563eb;
    }
    .nlp-badge {
        display: inline-block;
        background-color: #dbeafe;
        color: #1d4ed8;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
    .budget-badge {
        display: inline-block;
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
    .luxury-badge {
        display: inline-block;
        background-color: #f3e8ff;
        color: #7e22ce;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
    .schedule-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATASET LOADER
# ==============================================================================
@st.cache_data
def get_cached_dataset():
    """Loads local dataset once and caches in memory."""
    try:
        return load_travel_dataset()
    except Exception as e:
        st.error(f"⚠️ Error loading travel dataset: {e}")
        return pd.DataFrame()


# ==============================================================================
# MATPLOTLIB EXPENSE VISUALIZATIONS
# ==============================================================================
def plot_expense_charts(budget_data: dict, opt_data: dict = None):
    """
    Renders clean, legible Matplotlib charts:
    1. Pie chart for percentage distribution
    2. Grouped bar chart comparing original vs. optimized costs
    """
    categories = ["Transport", "Stay", "Food", "Activities", "Other"]
    costs = [
        budget_data["transport_cost"],
        budget_data["accommodation_cost"],
        budget_data["food_cost"],
        budget_data["activity_cost"],
        budget_data["other_cost"]
    ]
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#64748b"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor('#ffffff')

    # 1. Donut / Pie Chart
    valid_cats = [c for c, v in zip(categories, costs) if v > 0]
    valid_costs = [v for v in costs if v > 0]
    valid_colors = [colors[i] for i, v in enumerate(costs) if v > 0]

    wedges, texts, autotexts = ax1.pie(
        valid_costs,
        labels=valid_cats,
        autopct='%1.1f%%',
        startangle=140,
        colors=valid_colors,
        wedgeprops=dict(width=0.48, edgecolor='white', linewidth=2),
        textprops=dict(color='#1e293b', fontsize=10, weight='bold')
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color('#ffffff')

    ax1.set_title("Expense Distribution (%)", fontsize=12, weight='bold', pad=12, color='#1e293b')

    # 2. Bar Chart
    x = np.arange(len(categories))
    bar_width = 0.35

    if opt_data and opt_data.get("needed_optimization"):
        opt_b = opt_data["optimized_breakdown"]
        opt_costs = [
            opt_b["transport_cost"],
            opt_b["accommodation_cost"],
            opt_b["food_cost"],
            opt_b["activity_cost"],
            opt_b["other_cost"]
        ]

        rects1 = ax2.bar(x - bar_width/2, costs, bar_width, label='Original Cost', color='#ef4444', alpha=0.85)
        rects2 = ax2.bar(x + bar_width/2, opt_costs, bar_width, label='Optimized Cost', color='#10b981', alpha=0.9)
        ax2.set_ylabel('Cost in ₹ (INR)', fontsize=10, color='#1e293b')
        ax2.set_title('Category Comparison: Original vs. Optimized', fontsize=12, weight='bold', pad=12, color='#1e293b')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, fontsize=9.5)
        ax2.legend(frameon=True, facecolor='#ffffff')
        ax2.grid(axis='y', linestyle='--', alpha=0.4)

        for r in rects1:
            h = r.get_height()
            ax2.annotate(f'₹{int(h):,}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2),
                         textcoords="offset points", ha='center', va='bottom', fontsize=8, color='#b91c1c')
        for r in rects2:
            h = r.get_height()
            ax2.annotate(f'₹{int(h):,}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2),
                         textcoords="offset points", ha='center', va='bottom', fontsize=8, color='#047857')
    else:
        rects = ax2.bar(categories, costs, color=colors, width=0.5, edgecolor='none')
        ax2.set_ylabel('Cost in ₹ (INR)', fontsize=10, color='#1e293b')
        ax2.set_title('Expense Breakdown by Category', fontsize=12, weight='bold', pad=12, color='#1e293b')
        ax2.grid(axis='y', linestyle='--', alpha=0.4)

        for r in rects:
            h = r.get_height()
            ax2.annotate(f'₹{int(h):,}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2),
                         textcoords="offset points", ha='center', va='bottom', fontsize=9, weight='bold', color='#1e293b')

    plt.tight_layout()
    return fig


# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.markdown("### ✈️ Smart Trip Planner")
nav_choice = st.sidebar.radio(
    "Choose View:",
    ["🧳 Trip Planner (Live Demo)", "📍 Browse 25 Destinations", "ℹ️ Viva Guide & Modules"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**BE Computer Engineering**  
*NLP Mini Project*  
- **NLP:** Stopwords + Tokenizer + Rules + TF-IDF Cosine Sim  
- **Dataset:** 25 Indian Destinations  
- **Operation:** 100% Offline (No paid APIs)
""")


# ==============================================================================
# VIEW 1: MAIN LIVE DEMO (5-STEP FLOW)
# ==============================================================================
if nav_choice == "🧳 Trip Planner (Live Demo)":
    st.markdown("""
    <div class="hero-header">
        <h1>✈️ Smart Budget Trip Planner</h1>
        <p>Plan Smarter. Travel Better. Stay Within Your Budget.</p>
    </div>
    """, unsafe_allow_html=True)

    df_destinations = get_cached_dataset()
    dest_options = ["Any / Let AI Recommend"] + sorted(df_destinations["Destination"].tolist()) if not df_destinations.empty else ["Any / Let AI Recommend"]

    # Quick demo preset buttons
    st.markdown("##### 🚀 Quick Demo Presets (Click to autofill):")
    p1, p2, p3 = st.columns(3)
    if p1.button("🏖️ Sample 1: Goa Beach Budget (Viva Test Case)"):
        st.session_state["f_dest"] = "Goa"
        st.session_state["f_travelers"] = 2
        st.session_state["f_days"] = 4
        st.session_state["f_budget"] = 20000
        st.session_state["f_prefs"] = ["Beach", "Adventure", "Food"]
        st.session_state["f_query"] = "I want a cheap beach trip with good food and some adventure."
        st.rerun()

    if p2.button("🏔️ Sample 2: Manali Adventure Trek"):
        st.session_state["f_dest"] = "Manali"
        st.session_state["f_travelers"] = 2
        st.session_state["f_days"] = 5
        st.session_state["f_budget"] = 26000
        st.session_state["f_prefs"] = ["Adventure", "Nature"]
        st.session_state["f_query"] = "Looking for an adventurous mountain trek with snow and valley views."
        st.rerun()

    if p3.button("👑 Sample 3: Jaipur Royal Palaces"):
        st.session_state["f_dest"] = "Jaipur"
        st.session_state["f_travelers"] = 3
        st.session_state["f_days"] = 3
        st.session_state["f_budget"] = 18000
        st.session_state["f_prefs"] = ["Historical", "Food", "Shopping"]
        st.session_state["f_query"] = "Affordable royal palace tour with authentic food and bazaar shopping."
        st.rerun()

    # --------------------------------------------------------------------------
    # 1. USER INPUTS
    # --------------------------------------------------------------------------
    st.markdown('<div class="step-banner">1️⃣ Step 1 — Enter Trip Details & Natural Language Request</div>', unsafe_allow_html=True)

    with st.form("main_trip_form"):
        col1, col2 = st.columns(2)

        with col1:
            default_dest = st.session_state.get("f_dest", "Goa")
            d_idx = dest_options.index(default_dest) if default_dest in dest_options else 0
            in_destination = st.selectbox(
                "📍 Destination Preference:",
                dest_options,
                index=d_idx,
                help="Choose a specific destination or 'Any / Let AI Recommend'."
            )

            in_travelers = st.number_input(
                "👥 Number of Travelers (People):",
                min_value=1,
                max_value=20,
                value=int(st.session_state.get("f_travelers", 2)),
                step=1
            )

            in_days = st.number_input(
                "🗓️ Trip Duration (Days):",
                min_value=1,
                max_value=20,
                value=int(st.session_state.get("f_days", 4)),
                step=1
            )

        with col2:
            in_budget = st.number_input(
                "💰 Available Budget (Total in ₹ INR):",
                min_value=1000,
                max_value=5000000,
                value=int(st.session_state.get("f_budget", 20000)),
                step=500
            )

            pref_options = [
                "Beach", "Adventure", "Nature", "Food", "Historical",
                "Religious", "Shopping", "Luxury", "Budget", "Relaxation"
            ]
            default_p = st.session_state.get("f_prefs", ["Beach", "Adventure", "Food"])
            in_prefs = st.multiselect(
                "🏷️ Travel Preferences (Categories):",
                pref_options,
                default=[p for p in default_p if p in pref_options]
            )

            default_q = st.session_state.get(
                "f_query",
                "I want a cheap beach trip with good food and some adventure."
            )
            in_nlp_query = st.text_area(
                "💬 Natural-Language Travel Request (Plain English):",
                value=default_q,
                placeholder="Example: I want a cheap beach trip with good food and some adventure.",
                height=70
            )

        submit_plan = st.form_submit_button("🚀 Generate My Trip Plan", type="primary", use_container_width=True)

    # PROCESS FORM
    if submit_plan:
        if in_budget <= 0:
            st.error("❌ Budget must be greater than zero.")
            st.stop()
        if in_travelers < 1 or in_days < 1:
            st.error("❌ Travelers and Days must be at least 1.")
            st.stop()

        with st.spinner("Processing NLP analysis and computing trip plan..."):
            # Step 2: NLP Analysis
            nlp_result = analyze_travel_request(in_nlp_query)

            # Step 3: Recommendations
            chosen_dest = None if in_destination.startswith("Any") else in_destination
            recommendations = get_recommendations(
                travelers=in_travelers,
                days=in_days,
                budget=in_budget,
                selected_destination=chosen_dest,
                manual_preferences=in_prefs,
                natural_query=in_nlp_query,
                top_n=3
            )

            target_dest = recommendations[0]
            if chosen_dest:
                for r in recommendations:
                    if r["Destination"].lower() == chosen_dest.lower():
                        target_dest = r
                        break

            # Step 4: Budget Calculation & Optimization
            budget_calc = calculate_trip_budget(
                target_dest["Raw_Data"],
                travelers=in_travelers,
                days=in_days,
                available_budget=in_budget
            )
            budget_opt = optimize_trip_budget(budget_calc, available_budget=in_budget)

            # Step 5: Day-wise Itinerary
            itinerary_plan = generate_day_wise_itinerary(
                destination=target_dest["Destination"],
                days=in_days,
                user_preferences=nlp_result["preferences"] + in_prefs,
                destination_data=target_dest["Raw_Data"]
            )

            st.session_state["trip_result"] = {
                "nlp_result": nlp_result,
                "recommendations": recommendations,
                "target_dest": target_dest,
                "budget_calc": budget_calc,
                "budget_opt": budget_opt,
                "itinerary_plan": itinerary_plan,
                "inputs": {
                    "destination": target_dest["Destination"],
                    "travelers": in_travelers,
                    "days": in_days,
                    "budget": in_budget,
                    "raw_query": in_nlp_query
                }
            }

    # --------------------------------------------------------------------------
    # RENDER THE 4 REMAINING STEPS ONCE GENERATED
    # --------------------------------------------------------------------------
    if "trip_result" in st.session_state:
        res = st.session_state["trip_result"]
        nlp_res = res["nlp_result"]
        recs = res["recommendations"]
        t_dest = res["target_dest"]
        b_calc = res["budget_calc"]
        b_opt = res["budget_opt"]
        itin = res["itinerary_plan"]
        inp = res["inputs"]

        st.markdown("---")

        # ----------------------------------------------------------------------
        # 2. NLP PREFERENCE ANALYSIS
        # ----------------------------------------------------------------------
        st.markdown('<div class="step-banner">2️⃣ Step 2 — NLP Preference & Intent Analysis</div>', unsafe_allow_html=True)
        
        nlp_c1, nlp_c2 = st.columns([2, 1])
        with nlp_c1:
            st.markdown(f"**Natural Language Input:** *\"{inp['raw_query']}\"*")
            detected_badges = "".join([f"<span class='nlp-badge'>📍 {p}</span>" for p in nlp_res["preferences"]])
            st.markdown(f"**Detected Preferences:** {detected_badges}", unsafe_allow_html=True)
            if nlp_res["keywords"]:
                st.caption(f"**Extracted Keywords:** {', '.join(nlp_res['keywords'])}")

        with nlp_c2:
            tone = nlp_res["budget_type"]
            tone_class = "budget-badge" if tone == "Budget" else ("luxury-badge" if tone == "Luxury" else "nlp-badge")
            st.markdown(f"**Detected Budget Intent:** <span class='{tone_class}'>💰 {tone}</span>", unsafe_allow_html=True)
            st.caption("Extracted via lexical budget indicators & stopword filtering.")

        # ----------------------------------------------------------------------
        # 3. RECOMMENDATIONS
        # ----------------------------------------------------------------------
        st.markdown('<div class="step-banner">3️⃣ Step 3 — Top Recommended Destinations</div>', unsafe_allow_html=True)
        st.write("Dynamic ranking based on **Preference Match (50%) + Budget Suitability (35%) + Trip Duration (15%)**:")

        rcols = st.columns(3)
        for i, r in enumerate(recs[:3]):
            with rcols[i]:
                top_badge = "dest-card-top" if i == 0 else ""
                st.markdown(f"""
                <div class="dest-card {top_badge}">
                    <h3 style="margin:0; color:#1e3a8a;">#{i+1} {r['Destination']}</h3>
                    <p style="margin:2px 0 8px 0; color:#64748b;"><small>{r['State']} • {r['Type']}</small></p>
                    <p style="font-size:1.3rem; margin:4px 0; font-weight:bold; color:#2563eb;">
                        Match: {r['Match_Percentage']}%
                    </p>
                    <p style="margin:4px 0;"><b>Estimated Cost:</b> ₹{r['Estimated_Total_Cost']:,.0f}</p>
                    <div style="background:#f1f5f9; padding:8px; border-radius:6px; margin-top:8px;">
                        <small><b>Why Recommended:</b><br>{r['Reason']}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # 4. BUDGET BREAKDOWN & STATUS
        # ----------------------------------------------------------------------
        st.markdown(f'<div class="step-banner">4️⃣ Step 4 — Budget Breakdown & Status ({t_dest["Destination"]})</div>', unsafe_allow_html=True)

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("🚆 Transport", f"₹{b_calc['transport_cost']:,.0f}")
        b2.metric("🏨 Stay", f"₹{b_calc['accommodation_cost']:,.0f}")
        b3.metric("🍛 Food", f"₹{b_calc['food_cost']:,.0f}")
        b4.metric("🎯 Activities", f"₹{b_calc['activity_cost']:,.0f}")
        b5.metric("🛍️ Other", f"₹{b_calc['other_cost']:,.0f}")

        st.markdown(f"""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px 18px; margin:10px 0;">
            <span style="font-size:1.15rem; font-weight:bold;">Total Estimated Cost: </span>
            <span style="font-size:1.4rem; font-weight:bold; color:#1e40af;">₹{b_calc['total_trip_cost']:,.0f}</span>
            <span style="color:#64748b; margin-left:14px;">(Available Budget: ₹{b_calc['available_budget']:,.0f})</span>
        </div>
        """, unsafe_allow_html=True)

        # Budget Status
        if b_calc["is_within_budget"]:
            st.success(f"""
            **{b_calc['status_message']}**  
            Remaining Surplus: **₹{b_calc['remaining_budget']:,.0f}**
            """)
        else:
            st.warning(f"""
            **{b_calc['status_message']}**  
            Exceeded Amount: **₹{b_calc['exceeded_amount']:,.0f}**
            """)

            # Budget Optimization
            if b_opt["needed_optimization"]:
                st.markdown("##### 💡 AI Budget Optimization:")
                o1, o2, o3 = st.columns(3)
                o1.metric("Original Cost", f"₹{b_opt['original_cost']:,.0f}")
                o2.metric("Optimized Cost", f"₹{b_opt['optimized_cost']:,.0f}")
                o3.metric("Amount Saved", f"₹{b_opt['amount_saved']:,.0f}", delta=f"-₹{b_opt['amount_saved']:,.0f}")

                if b_opt["fits_within_budget"]:
                    st.success(f"{b_opt['status_message']} (Surplus: ₹{b_opt['remaining_budget']:,.0f})")
                else:
                    st.warning(f"{b_opt['status_message']}")

                with st.expander("🔍 View Actionable Cost Saving Strategies"):
                    for s in b_opt["suggestions"]:
                        st.markdown(f"- {s}")

        # Expense Charts
        st.markdown("##### 📊 Expense Visualizations:")
        fig = plot_expense_charts(b_calc, b_opt if b_opt["needed_optimization"] else None)
        st.pyplot(fig)

        # ----------------------------------------------------------------------
        # 5. DAY-WISE ITINERARY
        # ----------------------------------------------------------------------
        st.markdown(f'<div class="step-banner">5️⃣ Step 5 — Day-Wise Travel Itinerary ({t_dest["Destination"]}, {itin["total_days"]} Days)</div>', unsafe_allow_html=True)
        st.markdown(f"**Local Cuisine to Savor:** *{itin['local_cuisine']}* | **Recommended Transit:** *{itin['recommended_transit']}*")

        for d in itin["days_itinerary"]:
            with st.expander(f"📍 {d['title']} — {d['focus']}", expanded=(d['day_number'] == 1)):
                st.markdown(f"""
                <div class="schedule-card">
                    <p><b>🌅 Morning:</b> {d['morning']}</p>
                    <p><b>☀️ Afternoon:</b> {d['afternoon']}</p>
                    <p><b>🌇 Evening:</b> {d['evening']}</p>
                    <p><b>🌙 Night:</b> {d['night']}</p>
                    <p><b>🍽️ Recommended Meal:</b> Try <i>{d['recommended_food']}</i></p>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("💡 View Budget Travel Pro-Tips"):
            for t in itin["budget_travel_tips"]:
                st.markdown(t)


# ==============================================================================
# VIEW 2: BROWSE DESTINATIONS DATASET
# ==============================================================================
elif nav_choice == "📍 Browse 25 Destinations":
    st.markdown("""
    <div class="hero-header">
        <h1>📍 Browse 25 Destinations Dataset</h1>
        <p>Curated Indian travel destinations covering Beach, Adventure, Heritage, and Nature.</p>
    </div>
    """, unsafe_allow_html=True)

    df_dest = get_cached_dataset()
    if not df_dest.empty:
        c1, c2 = st.columns(2)
        q = c1.text_input("🔍 Search destination or activity:", "")
        t = c2.selectbox("🏷️ Filter by Type:", ["All"] + sorted(df_dest["Type"].unique().tolist()))

        f_df = df_dest.copy()
        if q:
            f_df = f_df[f_df["Destination"].str.lower().str.contains(q.lower()) | f_df["Popular_Activities"].str.lower().str.contains(q.lower())]
        if t != "All":
            f_df = f_df[f_df["Type"] == t]

        st.caption(f"Showing {len(f_df)} of {len(df_dest)} destinations.")

        for _, row in f_df.iterrows():
            st.markdown(f"""
            <div class="dest-card">
                <h4 style="margin:0; color:#1e3a8a;">{row['Destination']}, {row['State']} ({row['Type']})</h4>
                <p style="margin:4px 0;"><small><b>Popular Activities:</b> {row['Popular_Activities']}</small></p>
                <p style="margin:2px 0;"><small>Stay: ₹{row['Accommodation_Per_Day']}/day • Food: ₹{row['Food_Per_Day']}/day • Activities: ₹{row['Activity_Cost_Per_Day']}/day • Base Transport: ₹{row['Transport_Cost']}</small></p>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# VIEW 3: VIVA GUIDE & MODULES
# ==============================================================================
elif nav_choice == "ℹ️ Viva Guide & Modules":
    st.markdown("""
    <div class="hero-header">
        <h1>ℹ️ Viva Reference & System Modules</h1>
        <p>BE Computer Engineering NLP Mini Project Architecture</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### How the Complete System Works (Viva Explanation)

    ```
    [ User Input & NL Query ]
                │
                ▼
    [ Step 2: NLP Module ]
    ├── Lowercasing & Punctuation Cleanup
    ├── Stopword Removal & Tokenization
    ├── Preference Categorization (Beach, Food, Adventure...)
    └── Budget Intent Classification (Budget vs. Luxury)
                │
                ▼
    [ Step 3: Recommendation Engine ]
    ├── Preference Match Score (50%) + TF-IDF Cosine Similarity
    ├── Budget Suitability Score (35%)
    └── Trip Duration Fit Score (15%)
                │
                ▼
    [ Step 4: Budget Accounting & Optimization ]
    ├── Dynamic Itemized Calculation (Transport, Stay, Food, Activities, Misc)
    ├── Budget Status (Within vs. Exceeded)
    └── Automatic Expense Reduction if Over Budget
                │
                ▼
    [ Step 5: Day-Wise Itinerary Synthesis ]
    └── Tailored Morning/Afternoon/Evening/Night Schedule with Local Food Tips
    ```

    ---
    ### Quick Viva Q&A:
    - **Q: Where is NLP used in this project?**  
      **A:** NLP parses the freeform natural language query (`nlp.py`). It lowercases and tokenizes text, eliminates common English stopwords, extracts keywords, maps them to travel preferences (*Beach, Food, Adventure*), classifies budget intent (*Budget* vs *Luxury*), and computes **TF-IDF Vector representations and Cosine Similarity** against destination profiles.
    - **Q: How does the recommendation system rank destinations?**  
      **A:** It computes a composite score:
        $$\\text{Final Score} = \\text{Preference Match (50\\%)} + \\text{Budget Suitability (35\\%)} + \\text{Duration Alignment (15\\%)}$$
        normalized between $0\\%$ and $100\\%$.
    - **Q: What happens if a trip exceeds the user's budget?**  
      **A:** The **Budget Optimization Module** (`budget.py`) automatically simulates realistic reductions (budget stays/homestays, self-guided free activities, authentic local eateries) and recalculates whether the optimized trip fits the budget.
    - **Q: Does this project require paid APIs or internet?**  
      **A:** No. It is 100% self-contained and operates completely offline using `travel_data.csv`, standard Python libraries, and Streamlit.
    """)
