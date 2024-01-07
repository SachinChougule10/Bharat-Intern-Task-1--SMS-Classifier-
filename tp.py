import streamlit as st

import pandas as pd

data = pd.read_csv('spam.txt', sep = '\t', header=None, names=["label", "sms"])
data.head()

import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

stopwords = nltk.corpus.stopwords.words('english')
punctuation = string.punctuation

print(stopwords[:5])
print(punctuation)

def pre_process(sms):
    remove_punct = "".join([word.lower() for word in sms if word not in punctuation])
    tokenize = nltk.tokenize.word_tokenize(remove_punct)
    remove_stopwords = [word for word in tokenize if word not in stopwords]
    return remove_stopwords

#adding a column to our data with our processed messages
data['processed'] = data['sms'].apply(lambda x: pre_process(x))

print(data['processed'].head())

def categorize_words():
    spam_words = []
    ham_words = []
    #handling messages associated with spam
    for sms in data['processed'][data['label'] == 'spam']:
        for word in sms:
            spam_words.append(word)
    #handling messages associated with ham
    for sms in data['processed'][data['label'] == 'ham']:
        for word in sms:
            ham_words.append(word)
    return spam_words, ham_words

spam_words, ham_words = categorize_words()

print(spam_words[:5])
print(ham_words[:5])

def predict(sms):
    spam_counter = 0
    ham_counter = 0
    #count the occurances of each word in the sms string
    for word in sms:
        spam_counter += spam_words.count(word)
        ham_counter += ham_words.count(word)
    print('***RESULTS***')
    #if the message is ham
    if ham_counter > spam_counter:
        accuracy = round((ham_counter / (ham_counter + spam_counter) * 100))
        print('messege is not spam, with {}% certainty'.format(accuracy))
        return "Not Spam"
    #if the message is equally spam and ham
    elif ham_counter == spam_counter:
        print('message could be spam')
        return "Cannot detect"
    #if the message is spam
    else:
        accuracy = round((spam_counter / (ham_counter + spam_counter)* 100))
        print('message is spam, with {}% certainty'.format(accuracy))
        return  "Spam"






st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    processed_input = pre_process(input_sms)
    result = predict(processed_input)
    st.header(result)