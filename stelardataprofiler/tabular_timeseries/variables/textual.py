import pandas as pd

import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
from spacy_language_detection import LanguageDetector, detect_langs, DetectorFactory
from ftlangdetect import detect
import pycountry


def describe_textual(series: pd.Series, var_dict: dict, var_name: str) -> dict:
    series = series.astype(str)
    # Used in language detection
    DetectorFactory.seed = 2023

    var_dict.update(
        {
            "ratio_uppercase": 0,
            "ratio_digits": 0,
            "ratio_special_characters": 0,
            "language_distribution": [],
            "num_chars_distribution": {},
            "num_words_distribution": {}
        }
    )

    num_chars = 0
    ratio_uppercase = 0
    ratio_digits = 0
    ratio_special_characters = 0
    num_chars_list = []
    num_words_list = []
    corpus_languages = dict()

    texts_list = series.dropna().to_list()

    for text in texts_list:
        if not pd.isnull(text):
            text_num_chars = len(text)
            num_chars += text_num_chars
            num_chars_list.append(text_num_chars)
            for c in text:
                if c.isupper():
                    ratio_uppercase += 1
                if c.isdigit():
                    ratio_digits += 1
                if not c.isalnum():
                    ratio_special_characters += 1

            words = nltk.word_tokenize(text.lower())
            words_count = 0
            for word in words:
                num_words_list.append(len(word))

            # Find number of sentences
            sentences = nltk.sent_tokenize(text)
            sentences_count = 0
            for sentence in sentences:
                sentences_count += 1

            # Find languages
            try:
                languages = detect_langs(text)

                for language in languages:
                    if pycountry.languages.get(alpha_2=language.lang) is not None:
                        lang = pycountry.languages.get(alpha_2=language.lang).name.lower()
                    else:
                        lang = 'english'

                    if lang not in corpus_languages:
                        corpus_languages[lang] = language.prob
                    else:
                        corpus_languages[lang] += language.prob

            except:
                language = detect(text)

                if pycountry.languages.get(alpha_2=language['lang']) is not None:
                    lang = pycountry.languages.get(alpha_2=language['lang']).name.lower()
                else:
                    lang = 'english'

                if lang not in corpus_languages:
                    corpus_languages[lang] = language['score']
                else:
                    corpus_languages[lang] += language['score']

    # Calculate language distribution in the corpus

    corpus_languages = {k: v / var_dict['count'] for k, v in corpus_languages.items()}
    total = sum(corpus_languages.values(), float(0)) * 100
    if total < 100:
        corpus_languages['unknown'] = (100 - total) / 100

    corpus_languages = dict(sorted(corpus_languages.items(), key=lambda item: item[1], reverse=True))

    for k, v in corpus_languages.items():
        var_dict['language_distribution'].append({'language': k, "percentage": v * 100})

    if num_chars != 0:
        var_dict['ratio_uppercase'] = ratio_uppercase / num_chars
        var_dict['ratio_digits'] = ratio_digits / num_chars
        var_dict['ratio_special_characters'] = ratio_special_characters / num_chars

    if len(num_chars_list) != 0:
        s = pd.Series(num_chars_list)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        var_dict['num_chars_distribution'] = {
            'name': var_name,
            'min': stats[3],
            'max': stats[9],
            'average': stats[1],
            'stddev': stats[2],
            'median': stats[6],
            'kurtosis': s.kurtosis(),
            'skewness': s.skew(),
            'variance': s.var(),
            'percentile10': stats[4],
            'percentile25': stats[5],
            'percentile75': stats[7],
            'percentile90': stats[8],
        }

    if len(num_words_list) != 0:
        s = pd.Series(num_words_list)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        var_dict['num_words_distribution'] = {
            'name': var_name,
            'min': stats[3],
            'max': stats[9],
            'average': stats[1],
            'stddev': stats[2],
            'median': stats[6],
            'kurtosis': s.kurtosis(),
            'skewness': s.skew(),
            'variance': s.var(),
            'percentile10': stats[4],
            'percentile25': stats[5],
            'percentile75': stats[7],
            'percentile90': stats[8],
        }

    return var_dict
