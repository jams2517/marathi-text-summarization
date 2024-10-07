from __future__ import print_function
import streamlit as st
import collections
import io
import math
import operator
import sys
import networkx as nx
from preprocess import cleanText

window = 10
numberofSentences = 6
nodeHash = {}
textRank = {}
sentenceDictionary = collections.defaultdict(dict)
size = 0
sentences = []


def generatepositionaldistribution():
    global nodeHash, sentenceDictionary, sentences, size
    positional_dictionary = collections.defaultdict(dict)
    count = 0
    for i in sentenceDictionary.keys():
        for j in range(0, len(sentenceDictionary[i])):
            count += 1
            position = float(count) / (float(size) + 1.0)
            positional_dictionary[i][j] = 1.0 / \
                (math.pi * math.sqrt(position * (1 - position)))
            word = sentenceDictionary[i][j]
            if word in nodeHash:
                if nodeHash[word] < positional_dictionary[i][j]:
                    nodeHash[word] = positional_dictionary[i][j]
            else:
                nodeHash[word] = positional_dictionary[i][j]


def textrank():
    '''
        Generates a graph based ranking model for the tokens
    :return: Keyphrases that are most relevant for generating the summary.
    '''
    global sentenceDictionary, nodeHash, textRank
    graph = nx.Graph()
    graph.add_nodes_from(nodeHash.keys())
    for i in sentenceDictionary.keys():
        for j in range(0, len(sentenceDictionary[i])):
            current_word = sentenceDictionary[i][j]
            next_words = sentenceDictionary[i][j + 1:j + window]
            for word in next_words:
                graph.add_edge(current_word, word, weight=(
                    nodeHash[current_word] + nodeHash[word]) / 2)
    textRank = nx.pagerank(graph, weight='weight')
    keyphrases = sorted(textRank, key=textRank.get, reverse=True)[:n]
    return keyphrases


def summarize(filePath, keyphrases, numberofSentences):
    '''
        Generates the summary and writes the summary to the file.
    :param filePath: path of file to be used for summarization.
    :param keyphrases: Extracted keyphrases
    :param numberofSentences: Number of sentences needed as a summary
    :output: Writes the summary to the file
    '''
    global textRank, sentenceDictionary, sentences
    sentenceScore = {}
    for i in sentenceDictionary.keys():
        position = float(i + 1) / (float(len(sentences)) + 1.0)
        positionalFeatureWeight = 1.0 / \
            (math.pi * math.sqrt(position * (1.0 - position)))
        sumKeyPhrases = 0.0
        for keyphrase in keyphrases:
            if keyphrase in sentenceDictionary[i]:
                sumKeyPhrases += textRank[keyphrase]
        sentenceScore[i] = sumKeyPhrases * positionalFeatureWeight
    sortedSentenceScores = sorted(sentenceScore.items(
    ), key=operator.itemgetter(1), reverse=True)[:numberofSentences]
    sortedSentenceScores = sorted(
        sortedSentenceScores, key=operator.itemgetter(0), reverse=False)
    print("\nSummary: ")
    summary = []
    arr = []
    # for keyphrase in keyphrases:
    #     print(keyphrase)
    # print(keyphrases)

    for i in range(0, len(sortedSentenceScores)):
        arr.append(sentences[sortedSentenceScores[i][0]])
    s = "".join(arr)
    # print(s)
    return (s)


def process(arg1):
    '''
    :param arg1: path to the file containing the text to be summarized
    :param arg2: Number of sentences to be extracted as summary
    :param arg3: size of the window to be used in the co-occurance
    '''
    arg2 = 5
    arg3 = 6
    global window, n, numberofSentences, textRank, sentenceDictionary, size, sentences
    if arg1 != None and arg2 != None and arg3 != None:
        sentenceDictionary, sentences, size = cleanText(arg1)
        window = int(arg3)
        numberofSentences = int(arg2)
        n = int(math.ceil(min(0.1 * size, 7 * math.log(size))))
        generatepositionaldistribution()
        keyphrases = textrank()
        t = summarize(arg1, keyphrases, numberofSentences)
        return (t)
    else:
        print("not enough parameters")


if __name__ == "__main__":
    # process(sys.argv[1])
    st.markdown("<h1 style='text-align: center;'>Text Summarization</h1>",
                unsafe_allow_html=True)
    uploaded_files = st.file_uploader('Upload text file', type=[
                                      'txt'], accept_multiple_files=False)
    if uploaded_files is not None:
     # To read file as bytes:
        #  bytes_data = uploaded_file.getvalue()
        #  st.write(bytes_data)
        bytes_data = uploaded_files.read().decode('utf-8')
        result = process(bytes_data)
        st.subheader("Input Text\n")
        st.markdown(
            f"<div style='text-align: justify;'>{bytes_data}</div>",
            unsafe_allow_html=True)
        st.subheader("Summarized text\n")
        st.markdown(
            f"<div style='text-align: justify;'>{result}</div>",
            unsafe_allow_html=True)
