"""
================================================================================
Spam Detector - Text Preprocessing Utilities
================================================================================
A single, importable module that holds the text preprocessing function used by
both the training pipeline AND the persisted model.

WHY A SEPARATE MODULE?
-----------------------
scikit-learn pickles the TfidfVectorizer with a *reference* to the preprocessor
function (by module + name), not a copy. When the model is loaded later, Python
must be able to import that function by its fully-qualified name. Keeping the
preprocessor in its own stable module (instead of `__main__` or `train_model`)
guarantees the pickle loads correctly from any context: training, the Streamlit
app, ad-hoc scripts, etc.
================================================================================
"""

import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure the NLTK stopword corpus is available (silent download on first use).
for _pkg in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

_STEMMER = PorterStemmer()
_STOP_WORDS = set(stopwords.words("english"))


def preprocess(text):
    """
    Clean a single SMS message:
      1. lowercase
      2. strip punctuation
      3. drop English stopwords
      4. stem remaining words
    Returns the cleaned string.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = [
        _STEMMER.stem(word)
        for word in text.split()
        if word not in _STOP_WORDS
    ]
    return " ".join(tokens)
