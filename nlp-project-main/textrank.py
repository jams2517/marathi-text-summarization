import random
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from preprocess import clean_text

# Extended synonym dictionary for Marathi words
synonym_dict = {
    "छान": ["सुंदर", "आकर्षक", "मनोहर"],
    "मोठा": ["भव्य", "विशाल", "प्रचंड"],
    "शांत": ["स्थिर", "शांतता", "शांतप्रिय"],
    "आनंद": ["समाधान", "सुख", "खुश"],
    "शिक्षक": ["गुरु", "अध्यापक", "मास्तर"],
    "विद्यार्थी": ["शिकणारा", "शाळकरी", "शिष्य"],
    "नदी": ["तलाव", "जळ", "पाणीप्रवाह"],
    "पाऊस": ["धार", "पावसाळा", "वर्षाव"],
    "मित्र": ["सखा", "सोबती", "दोस्त"],
    "शहर": ["नगरे", "महानगर", "सिटी"],
    "कुटुंब": ["घराणे", "परिवार", "घरचे"],
    "नोकरी": ["काम", "व्यवसाय", "रोजगार"],
    "शब्द": ["अक्षर", "वाक्य", "वाणी"],
    "भूक": ["खाण्याची इच्छा", "आहार", "तृष्णा"],
    "संघ": ["गट", "समूह", "टीम"],
    "ज्ञान": ["विद्या", "शिक्षा", "माहिती"],
    "स्वतंत्र": ["मुक्त", "स्वाधीन", "स्वावलंबी"],
    "देश": ["राष्ट्र", "प्रदेश", "भूमी"],
    "प्रेम": ["आसक्ती", "आकर्षण", "आदर"],
    "सूर्य": ["सूर्यनारायण", "भास्कर", "दिवाकर"],
    "विविधतापूर्ण": ["बहुविध", "बहुरंगी", "विविधतेने भरलेला"],
    "धर्म": ["आस्था", "पंथ"],
    "संस्कृती": ["सभ्यता", "परंपरा", "रीतीरिवाज"],
    "भाषा": ["बोली", "वाणी", "भाषाभ्यास"],
    "परंपरा": ["सदभिरुची", "रूढी", "धर्मप्रथा"],
    "राज्य": ["प्रदेश", "संपन्न क्षेत्र", "सत्ता"],
    "महाराष्ट्र": ["मराठवाडा", "मराठी राज्य"],
    "महत्त्वाचे": ["महान", "अत्यावश्यक", "गरजेचे"],
    "सांस्कृतिक": ["सांस्कृतीक", "सांस्कृत", "परंपरागत"],
    "समृद्धी": ["भरभराट", "वैभव", "संपत्ती"],
    "ऐतिहासिक": ["इतिहासिक", "पुरातन", "गौरवशाली"],
    "निसर्ग": ["प्रकृती", "पर्यावरण", "सृष्टी"],
    "सौंदर्य": ["आकर्षकता", "रूप", "लावण्य"],
    "शहर": ["महानगर", "नगरी", "स्थान"],
    "वर्दळीचे": ["गर्दीचे", "गर्दीने भरलेले", "गडबडगोंधळाचे"],
    "उद्योग": ["कामधंदा", "व्यवसाय", "आर्थिक कार्य"],
    "संस्था": ["संघटना", "संवस्था", "केंद्र"],
    "फिल्म उद्योग": ["चित्रपटसृष्टी", "चलचित्र जगत", "फिल्मी"],
    "प्रसिद्ध": ["जगप्रसिद्ध", "ख्यातनाम", "सुप्रसिद्ध"],
    "स्वप्न": ["स्वप्नपूर्ती", "आकांक्षा", "इच्छा"],
    "सण": ["उत्सव", "समारंभ", "धार्मिक उत्सव"],
    "धूमधडाक्यात": ["धडाकेबाज", "साजरे करणे", "उत्साहाने"],
    "लोककला": ["जनकला", "लोकगीते", "जनजीवन कला"],
    "नृत्य": ["डान्स", "नाच", "नृत्यकला"],
    "किल्ले": ["दुर्ग", "गड", "बालेकिल्ले"],
    "शिवाजी महाराज": ["छत्रपती शिवाजी", "स्वराज्य संस्थापक", "मराठ्यांचे राजे"],
    "स्वराज्य": ["स्वतंत्रता", "साम्राज्य", "स्वतंत्र राज्य"],
    "कृषी": ["शेती", "कृषिव्यवसाय", "शेतकाम"],
    "उत्पादन": ["निर्मिती", "उत्पत्ति", "तयारी"],
    "औद्योगिक": ["कारखानदारी", "उद्योगसंबंधी", "व्यवसायिक"],
    "शिक्षण": ["विद्याभ्यास", "शैक्षणिक", "विद्या"],
    "उच्च शिक्षण": ["उच्च विद्याभ्यास", "महाविद्यालयीन", "तांत्रिक शिक्षण"],
    "संशोधन": ["अभ्यास", "अनुसंधान", "शोधकार्य"],
    "पर्यावरण": ["सृष्टी", "निसर्ग", "प्रकृती"],
    "नद्या": ["प्रवाह", "जलमार्ग", "उपनद्यांची"],
    "सहकार्य": ["सहभाग", "मदत", "सहाय्य"],
    "आपुलकी": ["मायाळू", "ममत्व", "आपत्य"],
    "सामंजस्य": ["एकात्मता", "समजूत", "सौहार्द"],
    "विकास": ["प्रगती", "वाढ", "उन्नती"]
    # Add more words to extend this dictionary
}

def replace_with_synonyms(sentence, synonym_dict):
    words = sentence.split()
    new_sentence = []
    
    for word in words:
        # Check if the word exists in the synonym dictionary
        if word in synonym_dict:
            # Randomly choose a synonym from the list
            synonym = random.choice(synonym_dict[word])
            new_sentence.append(synonym)
        else:
            # If no synonym, keep the word as it is
            new_sentence.append(word)
    
    return " ".join(new_sentence)

def generate_graph(sentences):
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(sentences)
    similarity_matrix = count_matrix * count_matrix.T
    
    # Create graph from similarity matrix
    graph = nx.from_numpy_array(similarity_matrix.toarray())
    return graph

def text_rank_abstractive(file_content):
    summary_sentence_count = 5
    sentence_dict, sentences, size = clean_text(file_content)
    
    # Replace words in each sentence with synonyms for abstraction
    abstractive_sentences = [replace_with_synonyms(sentence, synonym_dict) for sentence in sentences]
    
    # Generate the graph from the sentences
    graph = generate_graph(abstractive_sentences)
    
    # Use PageRank algorithm to rank sentences
    page_rank = nx.pagerank(graph)
    
    # Sort sentences by their rank and select the top N sentences for the summary
    ranked_sentences = sorted(page_rank, key=page_rank.get, reverse=True)
    top_sentences = ranked_sentences[:summary_sentence_count]
    
    summary = "\n".join([abstractive_sentences[sentence_id] for sentence_id in top_sentences])
    
    return summary
