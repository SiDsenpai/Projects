import os
import re
from collections import Counter

def analyze_persona(filepath, artist_name):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().lower()
        words = re.findall(r'\b\w+\b', text)
        total_words = len(words)
        if total_words == 0:
            print(f"❌ {artist_name} empty data!")
            return
        unique_words = len(set(words))
        lexical_richness = (unique_words / total_words) * 100
        print(f"🎤 --- {artist_name} Persona Stats ---")
        print(f"Total Words Analyzed: {total_words}")
        print(f"Unique Words (Vocabulary Size): {unique_words}")
        print(f"Lexical Richness (Complexity Score): {lexical_richness:.2f}%")
        print("-" * 40)
        
    except FileNotFoundError:
        print(f"⚠️ File not found: {filepath}")

# Teeno artists ko run karte hain
analyze_persona("dataset/Kr$na_lyrics.txt", "KR$NA")
analyze_persona("dataset/seedhe_maut_lyrics.txt", "Seedhe Maut")
analyze_persona("dataset/siddharth_lyrics.txt", "Siddharth Agarwal")