# import libraries
import os
import json
import streamlit as st

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res




with st.sidebar:
    
    "FAB2018 PROGRAMI VE PREFABRİK BETONARME BİLGİ BANKASI."
    "FAB2018 VE PREFABRİK BETONARME SOHBET ROBOTU."
    "FABBOT KENDİSİNE ÖNCEDEN YÜKLENMİŞ FAB2018 VE PREFABRİK BETONARME HAKKINDAKİ DOKÜMANA GÖRE CEVAP VERİR."
    "GENEL AMAÇLI BİR SOHBET ROBOTU DEĞİLDİR."
    
   

  
st.title("💬 FABBOT YAPAY ZEKA")
st.caption("🚀 Bir SAMYAP İnşaat Mühendislik Yazılım Uygulamasıdır.")
if "fab2018_model" not in st.session_state:
    st.session_state["fab2018_model"] = "gemini-1.5-pro-preview-0409"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Sorunuz nedir?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
      
        with st.spinner("Düşünüyorum..."):
         response=chatbot_response(prompt)
         response1 = st.write(response)
         st.session_state.messages.append({"role": "assistant", "content": response})
      
    
    