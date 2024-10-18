from __future__ import print_function, unicode_literals
import collections
import copy
import io
import re
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK tokenizers if not already available
nltk.download('punkt')
nltk.download('punkt_tab')

# Define global variables
stopwords = set()
sentences = []
sentences_processing = []
sentence_dictionary = collections.defaultdict(dict)
stemWords = {}

def readStemWords():
    '''
    Reads the stem words list and prepares the stem words dictionary for later use.
    '''
    global stemWords
    with io.open("word_list_marathi.txt", encoding='utf-8') as textFile:
        for line in textFile:
            line = line.strip()
            if len(line) > 0:
                # Extract word and its stem/related words
                wordEndIndex = line.find(">")
                word = line[2:wordEndIndex]
                baseEndIndex = line.find("]")
                base = line[1:baseEndIndex].strip()
                line = line[baseEndIndex + 1:]
                stem = None
                if len(base) >= 0:
                    stemEndIndex = base.find('-')
                    if stemEndIndex > 0:
                        stem = base[:stemEndIndex]

                valid = line[line.find("(") + 1: line.find(")")].strip()
                if valid == "0":
                    continue
                line = line[line.find("{") + 1: line.find("}")].strip()
                related = []
                if len(line) > 0:
                    split = line.split(",")
                    for s in split:
                        related.append(s[:s.find("|")])
                if stem is None and len(related) > 0:
                    stem = related[0]
                if stem is not None:
                    stemWords[word] = {"stem": stem, "related": related}

def tokenize(file_content):
    '''
    Tokenizes sentences and words from the file content.
    :param file_content: content of the file to be processed (string or file-like object)
    '''
    global sentences, sentences_processing, sentence_dictionary

    # Tokenize sentences using NLTK's sentence tokenizer
    sentences = sent_tokenize(file_content)
    sentences_processing = copy.deepcopy(sentences)

    counter = 0
    for sentence in sentences_processing:
        sentence = sentence.strip()
        sentence = re.sub(r'[^\w\s]', ' ', sentence)  # Remove punctuation
        tokens = sentence.split()
        actualTokens = removeStopWords(tokens)  # Remove stopwords
        stemmedTokens = stemmerMarathi(actualTokens)  # Apply stemming
        sentence_dictionary[counter] = stemmedTokens
        counter += 1

def readStopWords():
    '''
    Reads the stopwords from the file and adds them to the stopwords set.
    '''
    with io.open("stopwords.txt", encoding='utf-8') as textFile:
        for line in textFile:
            word = line.lower().strip()
            stopwords.add(word)

def removeStopWords(wordlist):
    '''
    Removes stopwords from the given list of words.
    :param wordlist: list of words to be filtered
    :return: filtered list without stopwords
    '''
    return [word for word in wordlist if word not in stopwords]

def removeCase(word):
    '''
    Reduces the word to its stem by removing case suffixes.
    :param word: word to be reduced
    :return: stemmed word
    '''
    word_length = len(word)
    
    # Apply suffix rules (modify as needed for your specific case rules)
    if word_length > 5:
        suffix = "शया"
        if word.endswith(suffix):
            return word[:-len(suffix)]

    # Additional rules for other suffixes can be added here

    return word

def removeNoGender(word):
    '''
    Removes gender suffixes and other inflections based on the stem dictionary.
    :param word: word to be stemmed based on gender suffixes
    :return: stemmed word
    '''
    global stemWords
    if word in stemWords:
        return stemWords[word]["stem"]

    # Add rules for handling gender inflection (based on suffix length)

    return word

def stemmerMarathi(words):
    '''
    Applies stemming to a list of Marathi words.
    :param words: list of words to be stemmed
    :return: list of stemmed words
    '''
    return [removeNoGender(removeCase(word)) for word in words]

def clean_text(file_content):
    '''
    Tokenizes the text, removes stopwords, and stems the words.
    :param file_content: content of the file to be preprocessed (string or file-like object)
    :return: sentence dictionary, list of sentences, and the size of processed tokens
    '''
    global sentence_dictionary, sentences
    readStopWords()  # Load stopwords
    tokenize(file_content)  # Tokenize the content into sentences and words

    size = 0
    for i in range(len(sentence_dictionary)):
        size += len(sentence_dictionary[i])

    # Remove any empty sentences from the dictionary
    sentence_dictionary = {key: value for key, value in sentence_dictionary.items() if len(value) > 0}

    return sentence_dictionary, sentences, size

# Initialize stem words dictionary at the start
readStemWords()
