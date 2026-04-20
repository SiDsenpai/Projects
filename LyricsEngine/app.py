import streamlit as st
import google.generativeai as genai
from collections import Counter
import re

API_KEY = "AIzaSyBLBD7THjpEloowhM8YppCodotgmFPchkc"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-3-flash-preview')

# UI Styling and Headers
st.set_page_config(page_title="VocalVerse AI", page_icon="🎤", layout="wide")
st.title("🎤 VocalVerse AI: DHH Persona Engine")
st.subheader("Powered by Siddharth Agarwal")

# Signatures
signatures = {
    "Siddharth": ["saiyam", "states", "kala", "halogen", "astatine", "iodine", "hurricane", "ghost", "planes"],
    "KRSNA": ["dollar", "sign", "lag", "aur", "par", "ab", "raha", "ban", "name", "hum", "yeh", "inaka", "jab", "main", "aaj"],
    "Seedhe Maut": ["Mo", "bhai", "raha", "nahi", "hu", "hum", "par", "ab", "yeh", "ghar", "yaha", "vo", "jab", "ek", "har"]
}

# Tabs
tab1, tab2 = st.tabs(["📊 Persona Analyzer", "✍️ AI Ghostwriter (Verse Generator)"])

# --- TAB 1: Analyzer ---
with tab1:
    st.write("### 📝 Enter your lyrics/bars:")
    user_input = st.text_area("Type your verse here...", height=150, key="analyzer")

    if st.button("Analyze Persona"):
        if user_input:
            words = user_input.lower().split()
            word_count = len(words)
            unique_words = len(set(words))
            richness = (unique_words / word_count) * 100 if word_count > 0 else 0
            
            st.success("✅ Analysis Complete!")
            col1, col2 = st.columns(2)
            col1.metric("Total Words", word_count)
            col2.metric("Lexical Richness", f"{richness:.2f}%")
            
            st.markdown("---")
            if richness > 25:
                st.info("🔥 **Persona Match:** Siddharth Agarwal (High Vocabulary Density)")
            elif richness > 15:
                st.warning("⚡ **Persona Match:** Seedhe Maut (Balanced Flow)")
            else:
                st.error("🥶 **Persona Match:** KR$NA (Technical Repetition)")
        else:
            st.error("Enter Your lyrics!")

# --- TAB 2: TRUE GENERATOR ---
with tab2:
    st.write("### 🤖 The Generative AI Co-Writer")
    st.markdown("Enter your verse! We will give u the next 4 bars in your desired artist style")
    
    sim_input = st.text_area("Write your starting bars here...", height=100, key="generator")
    target_persona = st.selectbox("🎯 Kiski awaz mein verse complete karna hai?", ["Siddharth", "KRSNA", "Seedhe Maut"])
    
    if st.button("✨ Generate Next 4 Bars"):
        if sim_input:
            if API_KEY == "AIzaSyBLBD7THjpEloowhM8YppCodotgmFPchkc":
                st.error("⚠️ Paste your Google API Key")
            else:
                with st.spinner(f"{target_persona} style is loading..."):
                    prompt = f"""
                    You are a Desi Hip Hop (DHH) rapper. Your persona is {target_persona}. 
                    Here is the starting verse written by me:
                    "{sim_input}"
                    
                    Task: Write the NEXT 4 BARS to complete this verse. 
                    - Keep the flow, rhyme scheme, and aggressive attitude.
                    - STRICTLY try to use some of these signature words: {signatures[target_persona]}
                    - Only give me the 4 lines, nothing else. Language: Hinglish.
                    """
                    
                    try:
                        response = model.generate_content(prompt)
                        st.success(f"🔥 {target_persona} Ghostwriter Verse Generated!")
                        st.text_area("AI Generated Bars:", value=response.text, height=150)
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.error("Write Something!")