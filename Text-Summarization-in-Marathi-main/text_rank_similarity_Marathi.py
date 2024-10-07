import streamlit as st
from preprocess import cleanText
import networkx
import itertools
import math
import sys
import io
import json
import gradio as gr


# Global dictionary to store sentence information
sentenceDictionary = {}

# Function to compute similarity between two sentences
def getSimilarity(sentenceID1, sentenceID2):
    commonWordCount = len(set(sentenceDictionary[sentenceID1]) & set(
        sentenceDictionary[sentenceID2]))
    denominator = math.log(len(
        sentenceDictionary[sentenceID1])) + math.log(len(sentenceDictionary[sentenceID2]))
    return commonWordCount/denominator if denominator else 0

# Function to generate a similarity graph
def generateGraph(nodeList):
    graph = networkx.Graph()
    graph.add_nodes_from(nodeList)
    edgeList = list(itertools.product(nodeList, repeat=2))
    for edge in edgeList:
        graph.add_edge(edge[0], edge[1],
                       weight=getSimilarity(edge[0], edge[1]))
    return graph

# Function to print the sentence dictionary
def printDictionary():
    for key, val in sentenceDictionary.items():
        print(str(key) + " : " + " ".join(sentenceDictionary[key]))

# Text Rank Similarity function modified to handle file content directly
def textRankSimilarity(fileContent):
    global sentenceDictionary
    summarySentenceCount = 5
    sentenceDictionary = {}
    sentences = []
    # Pass the file content (string) to the cleanText function
    sentenceDictionary, sentences, size = cleanText(io.StringIO(fileContent))
    graph = generateGraph(list(sentenceDictionary.keys()))
    pageRank = networkx.pagerank(graph)
    output = "\n".join([sentences[sentenceID] for sentenceID in sorted(
        sorted(pageRank, key=pageRank.get, reverse=True)[:summarySentenceCount])])
    return (output)

if __name__ == "__main__":
    # Streamlit UI components
    st.markdown("<h1 style='text-align: center;'>Text Summarization</h1>",
                unsafe_allow_html=True)
    uploaded_files = st.file_uploader('Upload text file', type=[
                                      'txt'], accept_multiple_files=False)
    if uploaded_files is not None:
        # Read the uploaded file and decode it to string
        bytes_data = uploaded_files.read().decode('utf-8')
        # Call textRankSimilarity with file content instead of file path
        result = textRankSimilarity(bytes_data)
        # Display the input text and the summarized output
        st.subheader("Input Text\n")
        st.markdown(f"<div style='text-align: justify;'>{bytes_data}</div>",
                    unsafe_allow_html=True)
        st.subheader("Summarized text\n")
        st.markdown(
            f"<div style='text-align: justify;'>{result}</div>",
            unsafe_allow_html=True)

