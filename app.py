import gradio as gr
import tensorflow as tf 
import numpy as np
import librosa
import time
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

p = pipeline("automatic-speech-recognition")

def onlyEnglish(filename):
    model = tf.keras.models.load_model("models/English.hdf5")
    class_names = ['Anger','Anxious','Apologetic','Concerned','Encouraging','Excited','Happiness','Sadness']
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label=model.predict(mfccs_scaled_features)
    classes_x=np.argmax(predicted_label,axis=1)
    class_num = classes_x[0]
    return class_names[class_num]

def englishGerman(filename):
    model = tf.keras.models.load_model("models/English-German.hdf5")
    class_names = ['Anger','Happiness','Sadness']
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label=model.predict(mfccs_scaled_features)
    classes_x=np.argmax(predicted_label,axis=1)
    class_num = classes_x[0]
    return class_names[class_num]

def multiple(filename):
    model = tf.keras.models.load_model("models/Multiple.hdf5")
    class_names = ["Anger","Disgust","Fear","Happiness","Neutral","Sadness"]
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label=model.predict(mfccs_scaled_features)
    classes_x=np.argmax(predicted_label,axis=1)
    class_num = classes_x[0]
    return class_names[class_num]

#calculate the negative, positive, neutral and compound scores, plus verbal evaluation
def sentiment_vader(sentence):

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    if sentiment_dict['compound'] >= 0.05 :
        overall_sentiment = "Positive"

    elif sentiment_dict['compound'] <= - 0.05 :
        overall_sentiment = "Negative"

    else :
        overall_sentiment = "Neutral"

    return overall_sentiment

def transcribe(audio, state=""):
    time.sleep(3)
    text = p(audio)["text"]
    text = sentiment_vader(text)
    return text

with gr.Blocks() as interface:
    gr.Markdown("<h1 style='text-align: center'>Deployed 3 different Models for different types of languages.</h1><ol><li>Model 1 has 8 Classes of Emotions and is trained only on English Language. This model extracts features and trains a NN model upon that.</li><li>Model 2 has 3 Classes of Emotions and is trained only on English and German. This model extracts features and trains a NN model upon that.</li><li>Model 3 has 6 Classes of Emotions and is trained only on English, French, German and Mexican. This model extracts features and trains a NN model upon that.</li><li>Model 4 has 3 Classes of Emotions and is trained only on English. It converts Speech-to-Text and then performs Sentiment Analysis.</li></ol>")
    with gr.Tabs():
        with gr.TabItem("Only English Model"):
            english_input = gr.inputs.Audio(label="Input Audio", type="filepath")
            english_button = gr.Button("Submit")
            english_output = gr.outputs.Label(num_top_classes = 8)
            
        with gr.TabItem("English-German Model"):
            two_input = gr.inputs.Audio(label="Input Audio", type="filepath")
            two_button = gr.Button("Submit")
            two_output = gr.outputs.Label(num_top_classes = 3)
            
        with gr.TabItem("English-French-German-Mexican Model"):
            multiple_input = gr.inputs.Audio(label="Input Audio", type="filepath")
            multiple_button = gr.Button("Submit")
            multiple_output = gr.outputs.Label(num_top_classes = 6)
        
        with gr.TabItem("Speech to Text"):
            stt_input = gr.inputs.Audio(label="Input Audio", type="filepath")
            stt_button = gr.Button("Submit")
            stt_output = gr.outputs.Label(num_top_classes = 6)

    english_button.click(onlyEnglish, inputs=english_input, outputs=english_output)
    two_button.click(englishGerman, inputs=two_input, outputs=two_output)
    multiple_button.click(englishGerman, inputs=multiple_input, outputs=multiple_output)
    stt_button.click(transcribe, inputs=stt_input, outputs=stt_output)

interface.launch()

