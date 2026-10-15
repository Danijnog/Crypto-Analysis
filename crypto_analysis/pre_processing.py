import unicodedata
import fasttext
import re

from datetime import datetime

def extract_date(filename: str) -> datetime:
    """Extracts the date from a filename in the format YYYY-MM-DD."""
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    return datetime.strptime(match[0], "%Y-%m-%d")


def clean_text_lang_detection(text: str) -> str:
    """"Cleans the input text for language detection by removing URLs, punctuation, and extra whitespace, and converting to lowercase."""

    # Remove URLs links
    text = re.sub(r'http\S+|www\S+|discord\S+', '', text)

    # Remove all Unicode punctuation marks
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith(("P", "S"))) # Punctuation and Symbol categories

    # Collapse multiple spaces into a single space and strip leading/trailing whitespace
    text = " ".join(text.split())

    # Convert to lowercase and remove leading/trailing whitespace
    return text.lower().strip()


def predict_language_confidence(text: str, model: fasttext.FastText) -> tuple:
    """
    Predicts the language and confidence of the input text using the fastText model.
    Returns the predicted language label (e.g., 'en' for English) and the confidence.
    """

    # Replace newlines because fasttext model expects a single line per string
    cleaned_text = clean_text_lang_detection(text).replace("\n", " ")

    if not cleaned_text:
        return None

    predictions = model.predict(cleaned_text, k=1)
    
    predicted_label = predictions[0][0]  # e.g., '__label__en'
    confidence = predictions[1][0]        # e.g., 0.98
    
    # Remove the '__label__' prefix to get just the language code
    return (predicted_label.replace("__label__", ""), confidence)


def get_lang_confidence(text: str, model: fasttext.FastText, threshold: float = 0.8) -> tuple[str, float, bool]:
    """
    Predicts the language of the given text using the provided FastText model and returns the language code, 
    confidence score, and a boolean indicating if the text is English.
    If the text is non-english, we check the confidence score against the threshold to determine if it is likely non-english
    to prevent false negatives.
    """
    is_english = True
    prediction = predict_language_confidence(text, model)
    if prediction is None:
        return None, None, is_english
    
    lang, confidence = prediction
    
    is_english = False if (lang != "en" and confidence >= threshold) else True
    return lang, confidence, is_english