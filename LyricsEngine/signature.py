# pip3 install scikit-learn
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def get_text(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().lower()
    except:
        return ""

print("🧠 AI is analyzing Signature Vocabularies...\n")

# Load data
artists = ["KR$NA", "Seedhe Maut", "Siddharth Agarwal"]
files = ["dataset/Kr$na_lyrics.txt", "dataset/seedhe_maut_lyrics.txt", "dataset/siddharth_lyrics.txt"]

corpus = [get_text(f) for f in files]

# Removing Stopwords
custom_stopwords = ['hai', 'ke', 'se', 'ko', 'ki', 'toh', 'me', 'mein', 'tu', 'mai', 'tha', 'the', 'aur', 'na', 'ye', 'wo', 'jo', 'ka', 'ek', 'hi', 'kya', 'bhi', 'ab', 'the', 'is', 'a', 'to', 'in', 'it', 'and', 'i', 'of', 'for']

# TF-IDF Model Initialize
vectorizer = TfidfVectorizer(stop_words=custom_stopwords, max_df=0.8)
tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

# Signature Words
for i, artist in enumerate(artists):
    print(f"👑 --- {artist}'s Top 15 Signature Words ---")
    
    artist_scores = tfidf_matrix[i].toarray().flatten()
    top_indices = artist_scores.argsort()[-15:][::-1]
    signature_words = [(feature_names[idx], artist_scores[idx]) for idx in top_indices]
    
    words_only = [word[0] for word in signature_words]
    print(", ".join(words_only))
    print("-" * 50)