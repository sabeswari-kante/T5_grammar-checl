import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
from transformers import T5Tokenizer, TFT5ForConditionalGeneration


@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained('./eng')
    model = TFT5ForConditionalGeneration.from_pretrained('./eng')
    return tokenizer, model


tokenizer, model = load_model()


def grammar_correct(sentence):
    enc = tokenizer(sentence, return_tensors="tf")
    out = model.generate(**enc, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(out[0], skip_special_tokens=True)


st.title("English Grammar Correction")
st.write("Enter text below to correct grammar mistakes.")

text = st.text_input("Enter text here", placeholder="Type your sentence...")

if st.button("Correct Grammar"):
    if text.strip():
        with st.spinner("Correcting grammar..."):
            corrected_text = grammar_correct(text)

            st.subheader("Results:")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Input:**")
                st.info(text)

            with col2:
                st.write("**Corrected:**")
                st.success(corrected_text)
    else:
        st.warning("Please enter some text first!")