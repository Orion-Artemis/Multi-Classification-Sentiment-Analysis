import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import re


model_name = "OmRajesh/bert-multi-label-classisfication"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()

# Define a function to predict emotions
def predict_emotions(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Move inputs to the appropriate device
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Generate predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)


    # Decode the predictions
    logits = outputs.logits
    sigmoid_output = torch.sigmoid(logits)
    predictions = (sigmoid_output > 0.5).int()
    return predictions

# Function to split text into sentences and remove full stops
def preprocessTexts(text):
    # Split based on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    # Remove full stops from each sentence
    cleaned_sentences = [s.replace('.', '').strip() for s in sentences]
    return cleaned_sentences

def emotionList(preds):
    emotions = []
    mixed_tensors = []
    for i in predictions:
        mixed_tensors.append(torch.nonzero(i).squeeze())

    mixed_tensors_cpu = [t.cpu() for t in mixed_tensors]  # Move all tensors to CPU

    # Iterate through each tensor in the mixed list
    for tensor in mixed_tensors_cpu:
        # Check the number of dimensions
        if tensor.dim() == 1:  # 1-D tensor
            for value in tensor:
                emotions.append(value.item())
        elif tensor.dim() == 2:  # 2-D tensor
            for row in tensor:
                for value in row:
                    emotions.append(value.item())
        else:
            emotions.append(tensor.item())
    emotions = [emotions_list[i] for i in emotions]
    return emotions

# Streamlit app
st.title('Multi-Emotion Classification')
st.write('This application leverages a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model specifically fine-tuned for sequence classification tasks. The model, BertForSequenceClassification, is designed to analyze text input and classify it into predefined categories. In this case, the model has been fine-tuned to recognize and classify emotions from textual data. By loading the model from the bert_emotion_classifier directory, the app can process user-provided text and predict the underlying emotion, enabling various applications such as sentiment analysis, customer feedback evaluation, and more.')
st.write('Enter a sentence or paragraph to analyze the emotions expressed in it.')

# Text input
user_input = st.text_area('Enter text here:', height=200)

# Create a button to copy text to clipboard
# Sample text to be copied
sample_text = "I am absolutely thrilled about my recent promotion! However, I feel a twinge of sadness knowing that a colleague, who worked just as hard, was overlooked. Despite this disappointment, I cherish our friendship and am relieved that we can still support each other."

# Streamlit app title
st.title('Sample Text')

# Create a button to load the sample text
if st.button("Load Sample Text"):
    # Display the sample text
    st.write("Sample Text:")
    st.write(sample_text)


emotions_list = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

if st.sidebar.button('Analyze'):
    if user_input:
        texts = preprocessTexts(user_input)
        predictions = predict_emotions(texts)
        emotions = emotionList(predictions)
        emotions = list(dict.fromkeys(emotions))
        st.sidebar.write('Emotions expressed in the text:')
        st.sidebar.write(emotions)
    else:
        st.sidebar.write('Please enter some text to analyze.')
