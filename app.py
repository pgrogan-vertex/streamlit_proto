import numpy as np
import pandas as pd

import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer

MAX_LENGTH = 512

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

from transformers import TFAutoModelForSequenceClassification

model = TFAutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=6)

model.load_weights('aws_experiments/exp_01_out/exp_01_6epochs.keras')

def manual_pred(text, return_probs = False):
    tok = tokenizer.encode(text,
                     truncation = True,
                    padding = 'max_length',
                    max_length=512,
                    return_tensors = 'tf')
    pred = model.predict(tok, verbose=False)
    probs = tf.nn.softmax(pred.logits).numpy()[0]
    if return_probs:
        return probs
    else:
        return np.argmax(probs)

################# STREAMLIT ###################

# Title
st.title("HF Interface Proto")

# Text input
user_input = st.text_input("Enter some text:")

# Display input
if user_input:
    pred = manual_pred(user_input)
    st.write("You entered:", user_input)
    st.write("Model prediction:", pred)

