# ✈️ SMART BUDGET TRIP PLANNER
### AI & NLP-Powered Travel Planning, Budget Optimization, and Dynamic Itinerary Generation System
**BE Computer Engineering — NLP Mini Project**

---

## 📌 1. Project Overview & Description
The **Smart Budget Trip Planner** is an intelligent, offline AI/NLP-based travel engineering system designed to help users plan customized vacations tailored to their available budget, trip duration, number of travelers, and natural-language preferences.

Unlike generic travel platforms that merely list static destinations, this system parses unstructured English travel requests (e.g., *"I want a cheap beach trip with good food and some adventure"*), extracts semantic intent, calculates multi-criteria destination recommendation scores, performs dynamic itemized budget accounting, and automatically synthesizes day-wise itineraries with expense visual analytics.

---

## 🎯 2. Objectives
1. **Natural Language Understanding (NLU):** Parse freeform user descriptions to detect travel preferences and classify budget expectations without requiring external paid LLM APIs.
2. **Multi-Factor Recommendation System:** Dynamically rank travel destinations based on semantic preference match, budget suitability, duration alignment, and TF-IDF similarity.
3. **Dynamic Budget Accounting:** Break down travel expenses into realistic cost heads: Transportation, Accommodation, Food, Sightseeing & Activities, and Miscellaneous.
4. **Constraint-Based Budget Optimization:** Automatically propose realistic cost reductions when a trip exceeds the user's available financial ceiling.
5. **Context-Aware Itinerary Synthesis:** Programmatically generate structured day-by-day itineraries reflecting destination-specific attractions and activities.
6. **100% Offline & Accessible Execution:** Operate reliably with local datasets and standard open-source Python libraries.

---

## ✨ 3. Key Features
- **Intelligent NLP Query Parsing:** Lowercasing, punctuation stripping, tokenization, stopword filtering, preference tagging, and budget tone classification.
- **Scikit-Learn TF-IDF Semantic Matching:** Vectorizes user requirements and destination corpora to compute cosine similarity.
- **Top-3 / Top-5 Dynamic Destination Recommendations:** Includes match percentage and human-readable explanation strings.
- **Itemized Dynamic Budget Estimation:** Dynamic formulas computing costs based on traveler headcount and trip duration.
- **Budget Optimization Engine:** Automatic expense reduction strategies across stay, dining, and activities with before-vs-after savings breakdown.
- **Personalized Day-Wise Itineraries:** Structured schedules with morning, afternoon, evening, and night activities, local culinary dishes, and transit pointers.
- **Matplotlib Visualizations:** Clean Donut/Pie charts for cost distribution and grouped Bar charts comparing original vs. optimized costs.
- **Interactive Multi-Page Streamlit Dashboard:** Pages for Trip Planning, Destination Exploration, Budget Simulator, Itinerary Viewer, and Academic Architecture Docs.

---

## 💻 4. Technology Stack
- **Programming Language:** Python 3.9+
- **Web Interface:** Streamlit
- **Data Manipulation & Analysis:** Pandas, NumPy
- **NLP & Machine Learning:** Scikit-learn (`TfidfVectorizer`, `cosine_similarity`), Regular Expressions (`re`)
- **Data Visualization:** Matplotlib
- **Local Data Storage:** Comma-Separated Values (`travel_data.csv`)
- **API Dependencies:** None (100% offline, zero paid API keys needed)

---

## 📂 5. Project Structure
```text
smart-budget-trip-planner/
│
├── app.py               # Streamlit application UI, visual analytics, and multi-page routing
├── nlp.py               # NLP text preprocessing, tokenization, preference extraction, TF-IDF
├── recommender.py       # Recommendation scoring engine (Preference + Budget + Duration)
├── budget.py            # Dynamic cost estimation and automated budget optimization
├── itinerary.py         # Dynamic day-wise itinerary synthesis with culinary & transit tips
├── travel_data.csv      # Local dataset of 25 realistic Indian travel destinations
├── requirements.txt     # Python dependencies
├── README.md            # Comprehensive academic project documentation
│
└── assets/
    └── README.txt       # Asset notes and guidelines
```

---

## 📊 6. Dataset Description (`travel_data.csv`)
The dataset includes **25 diverse Indian destinations** representing distinct geographies and travel typologies:

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `Destination` | String | Name of the tourist destination (e.g., Goa, Manali, Jaipur) |
| `State` | String | Indian state or union territory |
| `Type` | String | Primary destination typology (Beach, Adventure, Historical, Nature, Food, etc.) |
| `Transport_Cost` | Float (₹) | Estimated base travel cost per traveler |
| `Accommodation_Per_Day` | Float (₹) | Estimated daily lodging cost per traveler |
| `Food_Per_Day` | Float (₹) | Estimated daily food and dining cost per traveler |
| `Activity_Cost_Per_Day` | Float (₹) | Estimated daily sightseeing and activity cost per traveler |
| `Other_Cost_Per_Day` | Float (₹) | Estimated daily miscellaneous/contingency cost per traveler |
| `Popular_Activities` | String | Semicolon-delimited famous attractions and activities |
| `Keywords` | String | Space-delimited semantic keywords for NLP matching |
| `Recommended_Days` | Integer | Ideal trip duration in days |

**Destinations covered:** Goa, Manali, Jaipur, Udaipur, Kerala, Mumbai, Delhi, Agra, Rishikesh, Ooty, Mysore, Darjeeling, Shimla, Pune, Lonavala, Mahabaleshwar, Amritsar, Varanasi, Hyderabad, Bengaluru, Gokarna, Andaman, Jaisalmer, Coorg, Nashik.

---

## 🔬 7. System Modules & Methodology

### Module 1: Natural Language Processing (`nlp.py`)
1. **Text Normalization:** Converts input query to lower-case and strips non-alphanumeric punctuation.
2. **Tokenization & Stopword Filtering:** Splits text into whitespace-delimited tokens and filters out high-frequency non-content words using an optimized English stopword lexicon.
3. **Keyword Extraction:** Retains content-bearing keywords.
4. **Preference Extraction:** Matches tokens against 10 domain lexicons (*Beach, Adventure, Nature, Food, Historical, Religious, Shopping, Luxury, Budget, Relaxation*).
5. **Budget Tone Classification:**
   - Detects cues such as `cheap`, `budget`, `affordable`, `low cost`, `economical` $\rightarrow$ Classified as **Budget**.
   - Detects cues such as `luxury`, `premium`, `expensive`, `5-star`, `lavish` $\rightarrow$ Classified as **Luxury**.
   - Defaults to **Moderate** if balanced or unspecified.
6. **Scikit-Learn TF-IDF & Cosine Similarity:**
   Computes Term Frequency-Inverse Document Frequency matrices across destination corpus and measures cosine similarity:
   $$\text{Sim}(Q, D) = \frac{\mathbf{v}_Q \cdot \mathbf{v}_D}{\|\mathbf{v}_Q\| \|\mathbf{v}_D\|}$$

### Module 2: Recommendation Engine (`recommender.py`)
Computes a dynamic composite score for each destination:
$$\text{Final Recommendation Score} = \text{Score}_{\text{pref}} + \text{Score}_{\text{budget}} + \text{Score}_{\text{duration}}$$
- **Preference Match Score ($0 - 50$ pts):**
  - Category overlap ratio: up to 30 points.
  - Normalized TF-IDF cosine similarity: up to 20 points.
- **Budget Suitability Score ($0 - 35$ pts):**
  - Assesses ratio $R = \text{Estimated Total Cost} / \text{Available Budget}$.
  - $R \le 1.0$: Full score (33–35 points).
  - $1.0 < R \le 1.45$: Optimizable score (22–28 points).
  - $R > 1.45$: Progressive deficit penalty.
- **Duration Suitability Score ($0 - 15$ pts):**
  - Penalizes deviation $| \text{Days}_{\text{requested}} - \text{Days}_{\text{recommended}} |$.
- **Normalization:** Clamped between $25\%$ and $98\%$ and sorted descending.

### Module 3: Dynamic Budget Estimation (`budget.py`)
Calculates itemized costs dynamically:
$$\text{Transportation} = \text{Transport\_Cost} \times N_{\text{travelers}}$$
$$\text{Accommodation} = \text{Accommodation\_Per\_Day} \times N_{\text{days}} \times N_{\text{travelers}}$$
$$\text{Food} = \text{Food\_Per\_Day} \times N_{\text{days}} \times N_{\text{travelers}}$$
$$\text{Activities} = \text{Activity\_Cost\_Per\_Day} \times N_{\text{days}} \times N_{\text{travelers}}$$
$$\text{Other} = \text{Other\_Cost\_Per\_Day} \times N_{\text{days}} \times N_{\text{travelers}}$$
$$\text{Total Trip Cost} = \text{Transportation} + \text{Accommodation} + \text{Food} + \text{Activities} + \text{Other}$$

**Evaluation:**
- If $\text{Total Trip Cost} \le \text{Budget}$:
  `✅ TRIP IS WITHIN YOUR BUDGET` $\rightarrow$ $\text{Remaining Budget} = \text{Budget} - \text{Total Cost}$.
- If $\text{Total Trip Cost} > \text{Budget}$:
  `⚠️ TRIP EXCEEDS YOUR BUDGET` $\rightarrow$ $\text{Exceeded Amount} = \text{Total Cost} - \text{Budget}$.

### Module 4: Budget Optimization Engine (`budget.py`)
If the trip exceeds budget, constraint-based reductions are simulated:
- **Accommodation:** Switch to budget hostels/guesthouses (~35% reduction).
- **Activities:** Self-guided trails and group discounts (~40% reduction).
- **Food:** Savor local eateries and street delicacies (~20% reduction).
- **Transport:** Advance booking/sleeper train options (~15% reduction).
- **Other:** Trim discretionary retail (~40% reduction).
- Recalculates $\text{Optimized Cost}$ and reports precise net savings.

### Module 5: Context-Aware Itinerary Generator (`itinerary.py`)
- Synthesizes day-by-day plans based on trip duration and destination activities.
- Day 1: Arrival, accommodation check-in, orientation, and welcome dinner.
- Days 2 to $N-1$: Signature highlights (scuba/trekking/palaces), cultural exploration, and cuisine tasting.
- Day $N$: Breakfast, souvenir shopping, luggage packing, check-out, and return journey.
- Delivers destination-specific food specialties and local transit advice.

---

## 🛠️ 8. Installation & Setup

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12 installed.

### Step 1: Clone or Open Project Directory
```bash
cd smart-budget-trip-planner
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 9. How to Run the Application
Execute the Streamlit application using:
```bash
streamlit run app.py
```
Open your browser at the displayed local URL (typically `http://localhost:8501`).

---

## 🧪 10. Verification & Test Case

### Test Input:
- **Destination:** Goa
- **Number of Travelers:** 2
- **Number of Days:** 4
- **Available Budget:** ₹20,000
- **Travel Preferences:** Beach, Adventure, Food
- **Natural Language Request:**
  `"I want a cheap beach trip with good food and some adventure."`

### Expected & Verified Output:
1. **NLP Extraction:**
   - Categories detected: `Beach`, `Adventure`, `Food`, `Budget`
   - Budget classification: `Budget`
2. **Recommendation Engine:**
   - Destination: `Goa` (Rank #1)
   - Match Score: `~98%`
   - Reason: `Matches your Beach, Adventure, Food preferences and fits within your budget with smart budget optimization.`
3. **Itemized Budget Breakdown:**
   - Transportation: ₹5,000
   - Accommodation: ₹9,600
   - Food: ₹6,400
   - Activities: ₹4,800
   - Miscellaneous: ₹2,400
   - Total Cost: ₹28,200
4. **Budget Status:**
   - Status: `⚠️ TRIP EXCEEDS YOUR BUDGET`
   - Exceeded Amount: ₹8,200
5. **Budget Optimization:**
   - Original Cost: ₹28,200
   - Optimized Cost: ₹19,650
   - Amount Saved: ₹8,550
   - Status: `✅ Optimized trip fits within your budget.` (Remaining Balance: ₹350)
6. **Itinerary Synthesis:**
   - Complete 4-day plan with Calangute water sports, Baga beach hopping, Goan Fish Curry, and scooter rental tips.
7. **Expense Visualization:**
   - Rendered Matplotlib Pie chart and grouped comparison Bar chart.

---

## 🔮 11. Future Scope
1. **Integration with Real-Time APIs:** Connect live flight, train (IRCTC), and hotel pricing APIs for real-time rates.
2. **GPS & Map Route Optimization:** Integrate Leaflet/OpenStreetMap to visually map daily sightseeing routes and reduce travel times.
3. **Multi-City Route Planner:** Enable multi-destination itineraries (e.g., Golden Triangle: Delhi-Agra-Jaipur).
4. **Multilingual NLP:** Expand natural language processing to support regional Indian languages (Hindi, Marathi, Tamil, etc.).
