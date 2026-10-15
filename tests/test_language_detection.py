from crypto_analysis.pre_processing import clean_text_lang_detection, predict_language_confidence, get_lang_confidence
import fasttext

model = fasttext.load_model("../models/lid.176.bin")

def test_clean_text_lang_detection():
    text = "  ひらがな! Hey! https://example.com\nNew sentence. 🌳	:( :( "
    cleaned = clean_text_lang_detection(text)

    assert cleaned == "ひらがな hey new sentence"
    assert "http" not in cleaned
    assert cleaned == cleaned.lower()


def test_predict_language_confidence():
    assert predict_language_confidence("", model) is None

    language, confidence = predict_language_confidence(
        "Hello, this is a simple English sentence.", model
    )
    assert language == "en"
    assert 0.0 <= confidence <= 1.0


def test_get_lang_confidence():
    text = "Hello, this is a simple English sentence."
    language, _, is_english = get_lang_confidence(text, model)

    assert language == "en"
    assert is_english == True

    text = "Bonjour, ceci est une phrase en français."
    language, _, is_english = get_lang_confidence(text, model)

    assert language == "fr"
    assert is_english == False

test_clean_text_lang_detection()
test_predict_language_confidence()
test_get_lang_confidence()
