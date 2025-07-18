import os
import re
import string
import math
import gensim
import pandas as pd
from datetime import datetime
import dateutil.parser
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from simplemma import lemmatize
import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector
from collections import Counter
import pycountry
from typing import Union, Tuple, List
from ..utils import write_to_json


def profile_single_text(my_file_path: str) -> dict:
    """
    This method performs profiling and generates a profiling dictionary for a text file that exists in the given path.

    :param my_file_path: the path to a text file.
    :type my_file_path: str
    :return: A dict which contains the results of the profiler for the text.
    :rtype: dict

    """

    # Used in language detection
    def __get_lang_detector(nlp, name):
        return LanguageDetector(seed=2023)

    # Calculate TermFrequency and generate a matrix
    def __create_tf_matrix(freq_matrix):
        tf_matrix = {}

        for sent, f_table in freq_matrix.items():
            tf_table = {}

            count_words_in_sentence = len(f_table)
            for word, count in f_table.items():
                tf_table[word] = count / count_words_in_sentence

            tf_matrix[sent] = tf_table

        return tf_matrix

    # Create a table for documents per words
    def __create_documents_per_words(freq_matrix):
        word_per_doc_table = {}

        for sent, f_table in freq_matrix.items():
            for word, count in f_table.items():
                if word in word_per_doc_table:
                    word_per_doc_table[word] += 1
                else:
                    word_per_doc_table[word] = 1

        return word_per_doc_table

    # Calculate IDF and generate a matrix
    def __create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
        idf_matrix = {}

        for sent, f_table in freq_matrix.items():
            idf_table = {}

            for word in f_table.keys():
                idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

            idf_matrix[sent] = idf_table

        return idf_matrix

    # Calculate TF-IDF and generate a matrix
    def __create_tf_idf_matrix(tf_matrix, idf_matrix):
        tf_idf_matrix = {}

        for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

            tf_idf_table = {}

            for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                        f_table2.items()):  # here, keys are the same in both the table
                tf_idf_table[word1] = float(value1 * value2)

            tf_idf_matrix[sent1] = tf_idf_table

        return tf_idf_matrix

    # Important Algorithm: score the sentences
    def __score_sentences(tf_idf_matrix) -> dict:
        """
        score a sentence by its word's TF
        Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
        :rtype: dict
        """

        sentenceValue = {}

        for sent, f_table in tf_idf_matrix.items():
            total_score_per_sentence = 0

            count_words_in_sentence = len(f_table)
            for word, score in f_table.items():
                total_score_per_sentence += score

            if count_words_in_sentence != 0:
                sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence
            else:
                sentenceValue[sent] = 0

        return sentenceValue

    # Find the threshold
    def __find_average_score(sentenceValue) -> int:
        """
        Find the average score from the sentence value dictionary
        :rtype: int
        """
        sumValues = 0
        for entry in sentenceValue:
            sumValues += sentenceValue[entry]

        # Average value of a sentence from original summary_text
        average = (sumValues / len(sentenceValue))

        return average

    # Important Algorithm: Generate the summary
    def __generate_summary(sentences, sentenceValue, threshold):
        sentence_count = 0
        summary = ''

        for sentence in sentences:
            if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= threshold:
                summary += " " + sentence
                sentence_count += 1

        return summary.strip()

    if os.path.isdir(my_file_path):
        print('The input is not a file!')
        return dict()

    filename = get_filename(my_file_path)

    profile_dict = {
        'analysis': {
            'title': 'Profiling Report',
            'date_start': '',
            'date_end': '',
            'duration': '',
            'filenames': [filename]
        },
        'table': {
            'profiler_type': 'Textual',
            'num_texts': 1,
            'num_words': 0,
            'num_sentences': 0,
            'num_distinct_words': 0,
            'num_characters': 0,
            'ratio_uppercase': 0,
            'ratio_digits': 0,
            'ratio_special_characters': 0,
            'language': '',
            'language_distribution': [],
            'sentiment': 0,
            'named_entities': [],
            'term_frequency': []

        },
        'variables': []
    }

    now = datetime.now()
    start_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_start'] = start_string

    with open(my_file_path, 'r+') as text:
        text_dict = {
            'name': '',
            'type': 'Text',
            'num_words': 0,
            'num_sentences': 0,
            'num_distinct_words': 0,
            'num_characters': 0,
            'ratio_uppercase': 0,
            'ratio_digits': 0,
            'ratio_special_characters': 0,
            'language': '',
            'language_distribution': [],
            'summary': '',
            'topics': [],
            'sentiment': 0,
            'named_entities': [],
            'term_frequency': [],
            'special_characters_distribution': [],
            'sentence_length_distribution': dict(),
            'word_length_distribution': dict(),
        }

        # key is a special character and how many times is has been found in the text
        special_chars = {}

        # add the length of each word in the list to be used in the calculation of word_length_distribution
        word_length_list = []

        # add the length of each sentence in the list to be used in the calculation of sentence_length_distribution
        sentence_length_list = []

        text_dict['name'] = get_filename(my_file_path)

        file_contents = text.read()
        file_contents = ' '.join(file_contents.split())
        string_encode = file_contents.encode("ascii", "ignore")
        file_contents = string_encode.decode()

        # Find number of words
        words = nltk.word_tokenize(file_contents.lower())
        words_count = 0
        for word in words:
            words_count += 1
            word_length_list.append(len(word))
        profile_dict['table']['num_words'] = words_count
        text_dict['num_words'] = words_count

        # Find number of sentences
        sentences = nltk.sent_tokenize(file_contents)
        sentences_count = 0
        for sentence in sentences:
            sentences_count += 1
            sentence_length_list.append(len(sentence))
        profile_dict['table']['num_sentences'] = sentences_count
        text_dict['num_sentences'] = sentences_count

        # Find Distinct/Unique words
        unique_words = sorted(set(words))
        unique_words_count = len(unique_words)
        # set_of_unique_words.update(unique_words)
        profile_dict['table']['num_distinct_words'] = unique_words_count
        text_dict['num_distinct_words'] = unique_words_count

        # Find number of characters
        numCharacters = len(file_contents)
        text_dict['num_characters'] = numCharacters
        profile_dict['table']['num_characters'] = numCharacters

        # ratio_uppercase, ratio_digits, ratio_special_characters
        ratioUppercase = 0
        ratioDigits = 0
        ratioSpecialChars = 0
        for c in file_contents:
            if c.isupper():
                ratioUppercase += 1
            if c.isdigit():
                ratioDigits += 1
            if not c.isalnum():
                ratioSpecialChars += 1
                if c not in special_chars:
                    special_chars[c] = 1
                else:
                    special_chars[c] += 1

        text_dict['ratio_uppercase'] = ratioUppercase / numCharacters
        text_dict['ratio_digits'] = ratioDigits / numCharacters
        text_dict['ratio_special_characters'] = ratioSpecialChars / numCharacters
        profile_dict['table']['ratio_uppercase'] = text_dict['ratio_uppercase']
        profile_dict['table']['ratio_digits'] = text_dict['ratio_digits']
        profile_dict['table']['ratio_special_characters'] = text_dict['ratio_special_characters']

        # Find languages
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            print('Downloading language model for the spaCy POS tagger\n'
                  "(don't worry, this will only happen once)")
            from spacy.cli import download
            download('en')
            nlp = spacy.load('en_core_web_sm')
        if not Language.has_factory("language_detector"):
            Language.factory("language_detector", func=__get_lang_detector)
        nlp.add_pipe('language_detector', last=True)
        doc = nlp(file_contents)

        languages = {}
        cleaned_text = ' '
        lemma_text = ' '
        freq_matrix = Counter()
        for i, sent in enumerate(doc.sents):
            if sent.text:
                sentence = sent.text
                if pycountry.languages.get(alpha_2=sent._.language['language']) is not None:
                    language = pycountry.languages.get(alpha_2=sent._.language['language']).name.lower()
                else:
                    language = 'english'
                length_sent = len(sentence)
                if language not in languages:
                    languages[language] = float(sent._.language[
                                                    'score'] * length_sent / sentences_count * numCharacters)
                else:
                    languages[language] += float(sent._.language[
                                                     'score'] * length_sent / sentences_count * numCharacters)

                # Clean the sentence using the detecting language
                # Punctuation Removal
                cleaned_sentence = sentence.lower()
                for val in string.punctuation:
                    if val not in "'":
                        if val in "-":
                            cleaned_sentence = cleaned_sentence.replace(val, " ")
                        else:
                            cleaned_sentence = cleaned_sentence.replace(val, "")
                cleaned_sentence = ' '.join(cleaned_sentence.split()).strip()

                words = cleaned_sentence.split()

                # Stopword Removal
                if language in stopwords.fileids():
                    stop_words = set(stopwords.words(language))
                    cleaned_words = [w for w in words if not w in stop_words]
                else:
                    cleaned_words = words

                # Stemming
                stemmed_words = []
                if language in list(SnowballStemmer.languages):
                    stemmer = SnowballStemmer(language=language)
                    for word in cleaned_words:
                        word = stemmer.stem(word)
                        stemmed_words.append(word)
                else:
                    stemmed_words = cleaned_words

                # Lemma
                lemmatized_words = []
                if pycountry.languages.get(name=language) is not None:
                    for word in cleaned_words:
                        word = lemmatize(word, pycountry.languages.get(name=language).alpha_2)
                        lemmatized_words.append(word)
                else:
                    lemmatized_words = cleaned_words

                # freq_matrix will be used in summary extraction
                freq_matrix[sentence[:15]] = dict(Counter(stemmed_words))

                # add stemmed sentence to the cleaned_text
                cleaned_sentence = " ".join(stemmed_words)
                cleaned_text += cleaned_sentence.strip()
                cleaned_text += ' '

                # lemmatized text will be used in topic extraction
                lemmatized_text = " ".join(lemmatized_words)
                lemma_text += lemmatized_text.strip()
                lemma_text += ' '

        # Normalize language percentages
        total = sum(languages.values(), float(0))
        n_languages = {k: v * 100 / total for k, v in languages.items()}
        languages = n_languages
        # Find language most used in the text
        text_dict['language'] = max(languages, key=languages.get)
        profile_dict['table']['language'] = text_dict['language']

        # calculate language_distribution where all languages have percentages based on the sentences each language was detected
        total = sum(languages.values(), float(0))
        unknown_language_perc = 100
        for k, v in languages.items():
            if total >= 100:
                new_v = v * 100 / total
                text_dict['language_distribution'].append(
                    {'name': text_dict['name'], 'language': k, "percentage": new_v})
                profile_dict['table']['language_distribution'].append({'language': k, "percentage": new_v})
            else:
                text_dict['language_distribution'].append({'name': text_dict['name'], 'language': k, "percentage": v})
                profile_dict['table']['language_distribution'].append({'language': k, "percentage": v})
                unknown_language_perc -= v

        # Summary Extraction
        if len(file_contents.replace(" ", "")) > 300:
            '''
            Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
            '''
            # Calculate TermFrequency and generate a matrix
            tf_matrix = __create_tf_matrix(freq_matrix)
            # creating table for documents per words
            count_doc_per_words = __create_documents_per_words(freq_matrix)

            '''
            Inverse document frequency (IDF) is how unique or rare a word is.
            '''
            # Calculate IDF and generate a matrix
            idf_matrix = __create_idf_matrix(freq_matrix, count_doc_per_words, sentences_count)

            # Calculate TF-IDF and generate a matrix
            tf_idf_matrix = __create_tf_idf_matrix(tf_matrix, idf_matrix)

            # Important Algorithm: score the sentences
            sentence_scores = __score_sentences(tf_idf_matrix)

            # Find the threshold
            threshold = __find_average_score(sentence_scores)

            # Important Algorithm: Generate the summary
            summary = __generate_summary(sentences, sentence_scores, 1.8 * threshold)
            if not summary:
                summary = __generate_summary(sentences, sentence_scores, threshold)
                text_dict['summary'] = summary
            else:
                text_dict['summary'] = summary
        else:
            text_dict['summary'] = file_contents

        # Topic Extraction
        corpus = [lemma_text.split(' ')]

        dic = gensim.corpora.Dictionary(corpus)
        bow_corpus = [dic.doc2bow(doc) for doc in corpus]

        lda_model = gensim.models.LdaModel(bow_corpus,
                                           num_topics=1,
                                           id2word=dic,
                                           passes=100,
                                           iterations=100,
                                           random_state=2023,
                                           alpha='asymmetric')

        text_dict['topics'] = list(
            [token for token, score in lda_model.show_topic(i, topn=10)] for i in
            range(0, lda_model.num_topics))[0]

        # Sentiment Analysis
        sia = SentimentIntensityAnalyzer()
        compound_score = sia.polarity_scores(file_contents)['compound']

        text_dict['sentiment'] = compound_score
        profile_dict['table']['sentiment'] = compound_score

        # Named Entity Extraction
        named_entities = {}
        for X in doc.ents:
            sentence = X.text
            for val in string.punctuation:
                if val not in "'":
                    if val in "-":
                        sentence = sentence.replace(val, " ")
                    else:
                        sentence = sentence.replace(val, "")
            sentence = ' '.join(sentence.split()).strip()

            named_entities[sentence] = X.label_

        for ne, neType in named_entities.items():
            text_dict['named_entities'].append({'named_entity': ne, "type": neType})
            profile_dict['table']['named_entities'].append({'named_entity': ne, "type": neType})

        # Term Frequency
        data_analysis = dict(
            sorted(nltk.FreqDist(nltk.word_tokenize(cleaned_text)).items(), key=lambda item: item[1], reverse=True))

        for term, v in data_analysis.items():
            text_dict['term_frequency'].append({'name': text_dict['name'], 'term': term, "count": v})
            profile_dict['table']['term_frequency'].append({'term': term, "count": v})

        # text_dict['term_frequency'] = data_analysis
        # profile_dict['table']['term_frequency'] = data_analysis

        # calculate special_characters_distribution (FrequencyDistr)
        for k, v in special_chars.items():
            text_dict['special_characters_distribution'].append({'name': text_dict['name'], 'type': k, "count": v})

        # calculate sentence_length_distribution
        s = pd.Series(sentence_length_list)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        text_dict['sentence_length_distribution'] = {
            'name': text_dict['name'],
            'count': stats[0],
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

        # calculate word_length_distribution
        s = pd.Series(word_length_list)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        text_dict['word_length_distribution'] = {
            'name': text_dict['name'],
            'count': stats[0],
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

        profile_dict['variables'].append(text_dict)

    now = datetime.now()
    end_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_end'] = end_string

    profile_dict['analysis']['duration'] = str(
        dateutil.parser.parse(profile_dict['analysis']['date_end']) - dateutil.parser.parse(
            profile_dict['analysis']['date_start']))

    return profile_dict


# ----------- MULTIPLE TEXTS -----------#
def profile_multiple_texts(my_file_paths: List[str]) -> dict:
    """
    This method performs profiling and generates a profiling dictionary for the text files that exist in the given paths.

    :param my_file_paths: a list of paths leading to text files.
    :type my_file_paths: List[str]
    :return: A dict which contains the results of the profiler for the texts.
    :rtype: dict

    """

    # Used in language detection
    def __get_lang_detector(nlp, name):
        return LanguageDetector(seed=2023)

    # Calculate TermFrequency and generate a matrix
    def __create_tf_matrix(freq_matrix):
        tf_matrix = {}

        for sent, f_table in freq_matrix.items():
            tf_table = {}

            count_words_in_sentence = len(f_table)
            for word, count in f_table.items():
                tf_table[word] = count / count_words_in_sentence

            tf_matrix[sent] = tf_table

        return tf_matrix

    # Create a table for documents per words
    def __create_documents_per_words(freq_matrix):
        word_per_doc_table = {}

        for sent, f_table in freq_matrix.items():
            for word, count in f_table.items():
                if word in word_per_doc_table:
                    word_per_doc_table[word] += 1
                else:
                    word_per_doc_table[word] = 1

        return word_per_doc_table

    # Calculate IDF and generate a matrix
    def __create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
        idf_matrix = {}

        for sent, f_table in freq_matrix.items():
            idf_table = {}

            for word in f_table.keys():
                idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

            idf_matrix[sent] = idf_table

        return idf_matrix

    # Calculate TF-IDF and generate a matrix
    def __create_tf_idf_matrix(tf_matrix, idf_matrix):
        tf_idf_matrix = {}

        for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

            tf_idf_table = {}

            for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                        f_table2.items()):  # here, keys are the same in both the table
                tf_idf_table[word1] = float(value1 * value2)

            tf_idf_matrix[sent1] = tf_idf_table

        return tf_idf_matrix

    # Important Algorithm: score the sentences
    def __score_sentences(tf_idf_matrix) -> dict:
        """
        score a sentence by its word's TF
        Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
        :rtype: dict
        """

        sentenceValue = {}

        for sent, f_table in tf_idf_matrix.items():
            total_score_per_sentence = 0

            count_words_in_sentence = len(f_table)
            for word, score in f_table.items():
                total_score_per_sentence += score

            if count_words_in_sentence != 0:
                sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence
            else:
                sentenceValue[sent] = 0

        return sentenceValue

    # Find the threshold
    def __find_average_score(sentenceValue) -> int:
        """
        Find the average score from the sentence value dictionary
        :rtype: int
        """
        sumValues = 0
        for entry in sentenceValue:
            sumValues += sentenceValue[entry]

        # Average value of a sentence from original summary_text
        average = (sumValues / len(sentenceValue))

        return average

    # Important Algorithm: Generate the summary
    def __generate_summary(sentences, sentenceValue, threshold):
        sentence_count = 0
        summary = ''

        for sentence in sentences:
            if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= threshold:
                summary += " " + sentence
                sentence_count += 1

        return summary.strip()

    profile_dict = {
        'analysis': {
            'title': 'Profiling Report',
            'date_start': '',
            'date_end': '',
            'duration': '',
            'filenames': []
        },
        'table': {
            'profiler_type': 'Textual',
            'num_texts': 0,
            'num_words': 0,
            'num_sentences': 0,
            'num_distinct_words': 0,
            'num_characters': 0,
            'ratio_uppercase': 0,
            'ratio_digits': 0,
            'ratio_special_characters': 0,
            'language': '',
            'language_distribution': [],
            'sentiment': 0,
            'sentiment_analysis': {
                'compound_mean': 0.0,
                'compound_levels': {
                    '(-1, -0.5)': 0,
                    '(-0.5, 0)': 0,
                    '(0, 0.5)': 0,
                    '(0.5, 1)': 0
                }
            },
            'term_frequency': []

        },
        'variables': []
    }

    now = datetime.now()
    start_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_start'] = start_string

    corpus_languages = dict()
    set_of_unique_words = set()
    dict_term_freq = dict()
    compound_scores = {
        '(-1, -0.5)': 0,
        '(-0.5, 0)': 0,
        '(0, 0.5)': 0,
        '(0.5, 1)': 0
    }

    for text_file in my_file_paths:
        filename = get_filename(text_file)
        profile_dict['analysis']['filenames'].append(filename)
        with open(text_file, 'r+') as text:
            text_dict = {
                'name': filename,
                'type': 'Text',
                'num_words': 0,
                'num_sentences': 0,
                'num_distinct_words': 0,
                'num_characters': 0,
                'ratio_uppercase': 0,
                'ratio_digits': 0,
                'ratio_special_characters': 0,
                'language': '',
                'language_distribution': [],
                'summary': '',
                'topics': [],
                'sentiment': 0,
                'named_entities': [],
                'term_frequency': [],
                'special_characters_distribution': [],
                'sentence_length_distribution': dict(),
                'word_length_distribution': dict(),
            }

            # key is a special character and how many times is has been found in the text
            special_chars = {}

            # add the length of each word in the list to be used in the calculation of word_length_distribution
            word_length_list = []

            # add the length of each sentence in the list to be used in the calculation of sentence_length_distribution
            sentence_length_list = []

            file_contents = text.read()
            file_contents = ' '.join(file_contents.split())
            string_encode = file_contents.encode("ascii", "ignore")
            file_contents = string_encode.decode()

            if file_contents:
                profile_dict['table']['num_texts'] += 1

                # Find number of words
                words = nltk.word_tokenize(file_contents.lower())
                words_count = 0
                for word in words:
                    words_count += 1
                    word_length_list.append(len(word))
                profile_dict['table']['num_words'] += words_count
                text_dict['num_words'] = words_count

                # Find number of sentences
                sentences = nltk.sent_tokenize(file_contents)
                sentences_count = 0
                for sentence in sentences:
                    sentences_count += 1
                    sentence_length_list.append(len(sentence))
                profile_dict['table']['num_sentences'] += sentences_count
                text_dict['num_sentences'] = sentences_count

                # Find Distinct/Unique words
                unique_words = sorted(set(words))
                unique_words_count = len(unique_words)
                set_of_unique_words.update(unique_words)
                text_dict['num_distinct_words'] = unique_words_count

                # Find number of characters
                numCharacters = len(file_contents)
                text_dict['num_characters'] = numCharacters
                profile_dict['table']['num_characters'] += numCharacters

                # ratio_uppercase, ratio_digits, ratio_special_characters
                ratioUppercase = 0
                ratioDigits = 0
                ratioSpecialChars = 0
                for c in file_contents:
                    if c.isupper():
                        ratioUppercase += 1
                    if c.isdigit():
                        ratioDigits += 1
                    if not c.isalnum():
                        ratioSpecialChars += 1
                        if c not in special_chars:
                            special_chars[c] = 1
                        else:
                            special_chars[c] += 1

                text_dict['ratio_uppercase'] = ratioUppercase / numCharacters
                text_dict['ratio_digits'] = ratioDigits / numCharacters
                text_dict['ratio_special_characters'] = ratioSpecialChars / numCharacters
                profile_dict['table']['ratio_uppercase'] += ratioUppercase
                profile_dict['table']['ratio_digits'] += ratioDigits
                profile_dict['table']['ratio_special_characters'] += ratioSpecialChars

                # Find languages
                try:
                    nlp = spacy.load('en_core_web_sm')
                except OSError:
                    print('Downloading language model for the spaCy POS tagger\n'
                          "(don't worry, this will only happen once)")
                    from spacy.cli import download
                    download('en')
                    nlp = spacy.load('en_core_web_sm')
                if not Language.has_factory("language_detector"):
                    Language.factory("language_detector", func=__get_lang_detector)
                nlp.add_pipe('language_detector', last=True)
                doc = nlp(file_contents)

                languages = {}
                cleaned_text = ''
                lemma_text = ''
                freq_matrix = Counter()
                for i, sent in enumerate(doc.sents):
                    if sent.text:
                        sentence = sent.text
                        if pycountry.languages.get(alpha_2=sent._.language['language']) is not None:
                            language = pycountry.languages.get(alpha_2=sent._.language['language']).name.lower()
                        else:
                            language = 'english'
                        length_sent = len(sentence)
                        if language not in languages:
                            languages[language] = float(sent._.language[
                                                            'score'] * length_sent / sentences_count * numCharacters)
                        else:
                            languages[language] += float(sent._.language[
                                                             'score'] * length_sent / sentences_count * numCharacters)

                        # Clean the sentence using the detecting language
                        # Punctuation Removal
                        cleaned_sentence = sentence.lower()
                        for val in string.punctuation:
                            if val not in "'":
                                if val in "-":
                                    cleaned_sentence = cleaned_sentence.replace(val, " ")
                                else:
                                    cleaned_sentence = cleaned_sentence.replace(val, "")
                        cleaned_sentence = ' '.join(cleaned_sentence.split()).strip()

                        words = cleaned_sentence.split()

                        # Stopword Removal
                        if language in stopwords.fileids():
                            stop_words = set(stopwords.words(language))
                            cleaned_words = [w for w in words if not w in stop_words]
                        else:
                            cleaned_words = words

                        # Stemming
                        stemmed_words = []
                        if language in list(SnowballStemmer.languages):
                            stemmer = SnowballStemmer(language=language)
                            for word in cleaned_words:
                                word = stemmer.stem(word)
                                stemmed_words.append(word)
                        else:
                            stemmed_words = cleaned_words

                        # Lemma
                        lemmatized_words = []
                        if pycountry.languages.get(name=language) is not None:
                            for word in cleaned_words:
                                word = lemmatize(word, pycountry.languages.get(name=language).alpha_2)
                                lemmatized_words.append(word)
                        else:
                            lemmatized_words = cleaned_words

                        # freq_matrix will be used in summary extraction
                        freq_matrix[sentence[:15]] = dict(Counter(stemmed_words))

                        # add stemmed sentence to the cleaned_text
                        cleaned_sentence = " ".join(stemmed_words)
                        cleaned_text += cleaned_sentence.strip()
                        cleaned_text += ' '

                        # lemmatized text will be used in topic extraction
                        lemmatized_text = " ".join(lemmatized_words)
                        lemma_text += lemmatized_text.strip()
                        lemma_text += ' '

                # Normalize language percentages
                total = sum(languages.values(), float(0))
                n_languages = {k: v * 100 / total for k, v in languages.items()}
                languages = n_languages

                # Add languages dictionary to the corpus dictionary
                if corpus_languages is not {}:
                    corpus_languages = dict(Counter(corpus_languages) + Counter(languages))
                else:
                    corpus_languages = languages

                # Find language most used in the text
                text_dict['language'] = max(languages, key=languages.get)

                # calculate language_distribution where all languages have percentages based on the sentences each language was detected
                total = sum(languages.values(), float(0))
                unknown_language_perc = 100
                for k, v in languages.items():
                    if total >= 100:
                        new_v = v * 100 / total
                        text_dict['language_distribution'].append(
                            {'name': text_dict['name'], 'language': k, "percentage": new_v})
                    else:
                        text_dict['language_distribution'].append(
                            {'name': text_dict['name'], 'language': k, "percentage": v})
                        unknown_language_perc -= v

                # Summary Extraction
                if len(file_contents.replace(" ", "")) > 300:
                    '''
                    Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
                    '''
                    # Calculate TermFrequency and generate a matrix
                    tf_matrix = __create_tf_matrix(freq_matrix)
                    # creating table for documents per words
                    count_doc_per_words = __create_documents_per_words(freq_matrix)

                    '''
                    Inverse document frequency (IDF) is how unique or rare a word is.
                    '''
                    # Calculate IDF and generate a matrix
                    idf_matrix = __create_idf_matrix(freq_matrix, count_doc_per_words, sentences_count)

                    # Calculate TF-IDF and generate a matrix
                    tf_idf_matrix = __create_tf_idf_matrix(tf_matrix, idf_matrix)

                    # Important Algorithm: score the sentences
                    sentence_scores = __score_sentences(tf_idf_matrix)

                    # Find the threshold
                    threshold = __find_average_score(sentence_scores)

                    # Important Algorithm: Generate the summary
                    summary = __generate_summary(sentences, sentence_scores, 1.8 * threshold)
                    if not summary:
                        summary = __generate_summary(sentences, sentence_scores, threshold)
                        text_dict['summary'] = summary
                    else:
                        text_dict['summary'] = summary
                else:
                    text_dict['summary'] = file_contents

                # Topic Extraction
                corpus = [lemma_text.split(' ')]

                dic = gensim.corpora.Dictionary(corpus)
                bow_corpus = [dic.doc2bow(doc) for doc in corpus]

                lda_model = gensim.models.LdaModel(bow_corpus,
                                                   num_topics=1,
                                                   id2word=dic,
                                                   passes=100,
                                                   iterations=100,
                                                   random_state=2023,
                                                   alpha='asymmetric')

                text_dict['topics'] = list(
                    [token for token, score in lda_model.show_topic(i, topn=10)] for i in
                    range(0, lda_model.num_topics))[0]

                # Sentiment Analysis
                sia = SentimentIntensityAnalyzer()
                compound_score = sia.polarity_scores(file_contents)['compound']

                text_dict['sentiment'] = compound_score
                profile_dict['table']['sentiment'] += compound_score

                if compound_score > 0:
                    if compound_score >= 0.5:
                        compound_scores['(0.5, 1)'] += 1
                    else:
                        compound_scores['(0, 0.5)'] += 1
                elif compound_score < 0:
                    if compound_score <= -0.5:
                        compound_scores['(-1, -0.5)'] += 1
                    else:
                        compound_scores['(-0.5, 0)'] += 1

                profile_dict['table']['sentiment_analysis']['compound_mean'] += compound_score

                # Named Entity Extraction
                named_entities = {}
                for X in doc.ents:
                    sentence = X.text
                    for val in string.punctuation:
                        if val not in "'":
                            if val in "-":
                                sentence = sentence.replace(val, " ")
                            else:
                                sentence = sentence.replace(val, "")
                    sentence = ' '.join(sentence.split()).strip()

                    named_entities[sentence] = X.label_

                for ne, neType in named_entities.items():
                    text_dict['named_entities'].append({'named_entity': ne, "type": neType})

                # Term Frequency
                data_analysis = dict(
                    sorted(nltk.FreqDist(nltk.word_tokenize(cleaned_text)).items(), key=lambda item: item[1],
                           reverse=True))

                dict_term_freq = dict(Counter(dict_term_freq) + Counter(data_analysis))

                for term, v in data_analysis.items():
                    text_dict['term_frequency'].append({'term': term, "count": v})

                # calculate special_characters_distribution (FrequencyDistr)
                for k, v in special_chars.items():
                    text_dict['special_characters_distribution'].append(
                        {'name': text_dict['name'], 'type': k, "count": v})

                # calculate sentence_length_distribution
                s = pd.Series(sentence_length_list)
                stats = s.describe(percentiles=[.10, .25, .75, .90])

                text_dict['sentence_length_distribution'] = {
                    'name': text_dict['name'],
                    'count': stats[0],
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

                # calculate word_length_distribution
                s = pd.Series(word_length_list)
                stats = s.describe(percentiles=[.10, .25, .75, .90])

                text_dict['word_length_distribution'] = {
                    'name': text_dict['name'],
                    'count': stats[0],
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

                profile_dict['variables'].append(text_dict)

    # Calculate number of distinct words in the corpus
    profile_dict['table']['num_distinct_words'] = len(set_of_unique_words)

    # Calculate ratio_uppercase, ratio_digits, ratio_special_characters in the corpus
    profile_dict['table']['ratio_uppercase'] /= profile_dict['table']['num_characters']
    profile_dict['table']['ratio_digits'] /= profile_dict['table']['num_characters']
    profile_dict['table']['ratio_special_characters'] /= profile_dict['table']['num_characters']

    # Calculate language distribution in the corpus
    languages = {k: v / profile_dict['table']['num_texts'] for k, v in corpus_languages.items()}
    total = sum(languages.values(), float(0))
    unknown_language_perc = 100
    for k, v in languages.items():
        if total >= 100:
            new_v = v * 100 / total
            profile_dict['table']['language_distribution'].append({'language': k, "percentage": new_v})
        else:
            profile_dict['table']['language_distribution'].append({'language': k, "percentage": v})
            unknown_language_perc -= v

    if total < 100:
        profile_dict['table']['language_distribution'].append(
            {'language': "unknown", "percentage": unknown_language_perc})

    # Calculate Sentiment analysis for the corpus
    profile_dict['table']['sentiment'] /= profile_dict['table']['num_texts']
    profile_dict['table']['sentiment_analysis']['compound_levels'] = compound_scores
    profile_dict['table']['sentiment_analysis']['compound_mean'] /= profile_dict['table']['num_texts']

    # Calculate term frequency for the corpus
    data_analysis = dict(sorted(dict_term_freq.items(), key=lambda item: item[1], reverse=True))

    for term, v in data_analysis.items():
        profile_dict['table']['term_frequency'].append({'term': term, "count": v})

    now = datetime.now()
    end_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_end'] = end_string

    profile_dict['analysis']['duration'] = str(
        dateutil.parser.parse(profile_dict['analysis']['date_end']) - dateutil.parser.parse(
            profile_dict['analysis']['date_start']))

    return profile_dict


# ----------- MAIN FUNCTION ----------#
def profile_text_with_config(config: dict) -> None:
    """
    This method performs profiling on text data and writes the resulting profile dictionary based on a configuration dictionary.

    :param config: a dictionary with all configuration settings.
    :type config: dict
    :return: None.
    :rtype: None

    """
    # input file path(s)
    input_file_paths = config['input']['files']

    if isinstance(input_file_paths, list):
        if len(input_file_paths) == 1:
            my_path = os.path.abspath(input_file_paths[0])
        else:
            my_path = []
            for path in input_file_paths:
                my_path.append(os.path.abspath(input_file_paths))
    elif isinstance(input_file_paths, str) and os.path.isfile(os.path.abspath(input_file_paths)):
        my_path = os.path.abspath(input_file_paths)
    else:
        raise ValueError(f"Invalid input: {input_file_paths} must be a valid file path or list of file paths")

    # output file path
    output_json_path = os.path.abspath(config['output']['json'])

    # Run raster profile
    profile_dict = profile_text(my_path=my_path)

    # Write resulting profile dictionary
    write_to_json(profile_dict, output_json_path)


def profile_text(my_path: Union[str, List[str]]):
    """
    This method performs profiling and generates a profiling dictionary for either a single text or many texts.

    :param my_path: either the path to a text file or a list of paths to text files.
    :type my_path: Union[str, List[str]]
    :return: A dict which contains the results of the profiler for the text or texts.
    :rtype: dict

    """
    if isinstance(my_path, list):
        # Handle list of paths
        return profile_multiple_texts(my_path)
    elif isinstance(my_path, str) and os.path.isfile(my_path):
        # Handle single file path
        return profile_single_text(my_path)
    else:
        raise ValueError(f"Invalid input: {my_path} must be a valid file path or list of file paths")


def get_filename(path: str) -> Tuple[str, str]:
    """Helper to split filename and extension"""
    filename = os.path.basename(path)
    return filename
