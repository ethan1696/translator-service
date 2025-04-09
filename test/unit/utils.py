import spacy
from difflib import SequenceMatcher

# Load a small English model (lightweight)
nlp = spacy.load("en_core_web_sm")

def eval_single_response_complete(expected_answer: tuple[bool, str], llm_response: tuple[bool, str]) -> float:
    '''Evaluates whether both the language classification and the translated/original message are correct.'''
    expected_lang, expected_text = expected_answer
    actual_lang, actual_text = llm_response

    lang_score = 0.5 if expected_lang == actual_lang else 0.0

    expected_text_clean = expected_text.strip().lower()
    actual_text_clean = actual_text.strip().lower()

    if expected_text_clean == "unintelligible input.":
        text_score = 0.5 if actual_text_clean == "unintelligible input." else 0.0
    else:
        # Compute similarity using spaCy vectors (fallback to sequence similarity if not available)
        doc1 = nlp(expected_text_clean)
        doc2 = nlp(actual_text_clean)

        if doc1.vector_norm and doc2.vector_norm:
            similarity = doc1.similarity(doc2)
        else:
            similarity = SequenceMatcher(None, expected_text_clean, actual_text_clean).ratio()

        text_score = min(0.5, 0.5 * similarity)

    return lang_score + text_score
