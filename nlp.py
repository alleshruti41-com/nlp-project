"""
NLP Module for Smart Budget Trip Planner
=========================================
Performs natural language processing on user travel queries.
Features:
- Text lowercasing and punctuation removal
- Tokenization and stopword removal
- Keyword extraction
- Rule-based preference categorization
- Budget intent classification (Budget vs. Luxury vs. Moderate)
- Scikit-learn TF-IDF Vectorization and Cosine Similarity for semantic matching
"""

import re
import string
from typing import Dict, List, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Common English stopwords to filter out non-informative words
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "want", "like", "need",
    "looking", "plan", "go", "trip", "travel", "please", "would"
}

# Supported preference categories with indicator lexical sets
PREFERENCE_KEYWORDS = {
    "Beach": {
        "beach", "beaches", "sea", "ocean", "coastal", "coast", "sand", "sandy",
        "waves", "island", "islands", "shores", "shacks", "coastal", "sunbathing"
    },
    "Adventure": {
        "adventure", "adventurous", "trekking", "trek", "hike", "hiking", "rafting",
        "paragliding", "bungee", "camping", "camp", "safari", "scuba", "diving",
        "snorkeling", "thrill", "thrilling", "climbing", "water sports"
    },
    "Nature": {
        "nature", "greenery", "mountains", "mountain", "hills", "hill", "valley",
        "waterfalls", "waterfall", "wildlife", "forest", "tea", "lake", "mist",
        "scenic", "landscapes", "flora", "fauna", "coffee"
    },
    "Food": {
        "food", "foodie", "cuisine", "culinary", "dishes", "tasty", "delicious",
        "street food", "seafood", "dining", "biryani", "cafes", "eating", "eat",
        "local food", "delicacies"
    },
    "Historical": {
        "historical", "history", "heritage", "palace", "palaces", "fort", "forts",
        "monuments", "monument", "architecture", "ancient", "royal", "museum",
        "ruins", "dynasty", "culture"
    },
    "Religious": {
        "religious", "spiritual", "temple", "temples", "ghat", "ghats", "aarti",
        "darshan", "church", "pilgrimage", "holy", "monastery", "devotional",
        "prayer", "sacred"
    },
    "Shopping": {
        "shopping", "bazaar", "bazaars", "market", "markets", "souvenirs",
        "handicrafts", "clothes", "silk", "shopping street", "malls"
    },
    "Luxury": {
        "luxury", "luxurious", "premium", "expensive", "5-star", "five star",
        "resort", "resorts", "lavish", "posh", "fine dining", "high-end", "royal stay"
    },
    "Budget": {
        "cheap", "budget", "affordable", "low cost", "economical", "backpacker",
        "pocket friendly", "pocket-friendly", "student friendly", "inexpensive",
        "low budget", "saver"
    },
    "Relaxation": {
        "relaxation", "relax", "relaxing", "peaceful", "quiet", "calm", "serenity",
        "chill", "leisure", "spa", "ayurveda", "unwind", "peace", "tranquil", "soothing"
    }
}

# Specific budget vs luxury keyword cues for budget_type classification
BUDGET_KEYWORDS = {
    "cheap", "budget", "affordable", "low cost", "low-cost", "economical",
    "backpacker", "pocket friendly", "pocket-friendly", "inexpensive",
    "economical", "saving", "saver"
}

LUXURY_KEYWORDS = {
    "luxury", "luxurious", "premium", "expensive", "5-star", "five-star",
    "lavish", "high-end", "posh", "royal", "first class", "deluxe"
}


def clean_text(text: str) -> str:
    """
    1. Converts input to lowercase.
    2. Removes unnecessary punctuation and excess whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Replace hyphens in phrases like low-cost with space or keep for pattern
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # Remove punctuation except letters and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_text(cleaned_text: str) -> List[str]:
    """
    Tokenizes text into individual word tokens.
    """
    if not cleaned_text:
        return []
    return cleaned_text.split()


def extract_keywords(tokens: List[str]) -> List[str]:
    """
    Extracts informative keywords by filtering out general stopwords.
    """
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def classify_budget_type(raw_text: str) -> str:
    """
    Classifies user budget intent into 'Budget', 'Luxury', or 'Moderate'.
    """
    lower = raw_text.lower()
    
    # Check multi-word and single-word matches
    has_budget = any(w in lower for w in BUDGET_KEYWORDS)
    has_luxury = any(w in lower for w in LUXURY_KEYWORDS)

    if has_budget and not has_luxury:
        return "Budget"
    elif has_luxury and not has_budget:
        return "Luxury"
    elif has_budget and has_luxury:
        # If user says something like "budget luxury", prioritize balanced
        return "Moderate"
    else:
        return "Moderate"


def extract_preferences(raw_text: str) -> List[str]:
    """
    Extracts travel preference categories based on lexical indicators.
    Returns matched categories in order of appearance / relevance.
    """
    lower = raw_text.lower()
    detected_prefs = []

    for category, keywords in PREFERENCE_KEYWORDS.items():
        matched = False
        for kw in keywords:
            # Word boundary regex to avoid partial substring false matches
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, lower):
                matched = True
                break
        if matched:
            detected_prefs.append(category)

    return detected_prefs


def analyze_travel_request(raw_text: str) -> Dict[str, Any]:
    """
    Main NLP function.
    Accepts natural-language string and returns structured dictionary:
    {
        "raw_text": str,
        "clean_text": str,
        "tokens": list,
        "keywords": list,
        "preferences": list,
        "budget_type": str
    }
    """
    if not raw_text or not raw_text.strip():
        return {
            "raw_text": "",
            "clean_text": "",
            "tokens": [],
            "keywords": [],
            "preferences": [],
            "budget_type": "Moderate"
        }

    cleaned = clean_text(raw_text)
    tokens = tokenize_text(cleaned)
    keywords = extract_keywords(tokens)
    preferences = extract_preferences(raw_text)
    budget_type = classify_budget_type(raw_text)

    # Ensure "Budget" is in preferences if budget_type == "Budget" and not already added
    if budget_type == "Budget" and "Budget" not in preferences:
        preferences.append("Budget")
    elif budget_type == "Luxury" and "Luxury" not in preferences:
        preferences.append("Luxury")

    return {
        "raw_text": raw_text,
        "clean_text": cleaned,
        "tokens": tokens,
        "keywords": keywords,
        "preferences": preferences,
        "budget_type": budget_type
    }


def compute_tfidf_similarity(query_text: str, document_texts: List[str]) -> np.ndarray:
    """
    Scikit-learn TF-IDF Vectorization and Cosine Similarity computation.
    Compares the user query text against each destination's keyword/activity corpus.
    Returns array of similarity scores between 0.0 and 1.0.
    """
    if not query_text or not document_texts:
        return np.zeros(len(document_texts))

    cleaned_query = clean_text(query_text)
    cleaned_docs = [clean_text(str(doc)) for doc in document_texts]
    all_texts = [cleaned_query] + cleaned_docs

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        # Cosine similarity between query (index 0) and all documents (indices 1 to N)
        cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        max_val = float(np.max(cos_sim))
        min_val = float(np.min(cos_sim))
        if max_val > min_val and max_val > 0.05:
            normalized_sim = (cos_sim - min_val) / (max_val - min_val + 1e-9)
            return normalized_sim
        return cos_sim
    except Exception:
        # Fallback in case of empty vocabulary
        return np.zeros(len(document_texts))


if __name__ == "__main__":
    # Test sample query
    sample_text = "I want a cheap beach trip with good food and some adventure."
    result = analyze_travel_request(sample_text)
    print("NLP Analysis Result:")
    print("Clean Text:", result["clean_text"])
    print("Tokens:", result["tokens"])
    print("Keywords:", result["keywords"])
    print("Preferences:", result["preferences"])
    print("Budget Type:", result["budget_type"])
