

<h1 align="center">Multilingual Sentiment Analysis</h3>

<p align="center"> This project uses 4 separately trained models on different languages to perform Sentiment Analysis on any audio provided.
<ol>
<li><b>Model 1</b> - Trained on <b> English language </b>. Has an <b>Accuracy of 89%</b>. Extracts the features from uploaded audio and classifies into <b>8 different emotions </b> based on a custom trained Neural Network</li>
<br>
<li><b>Model 2</b> - Trained on <b>English and German</b>. Has an <b>Accuracy of 95%</b>. Extracts the features from uploaded audio and classifies into <b>3 different emotions</b> based on a custom trained Neural Network</li>
<br>
<li><b>Model 3</b> - Trained on <b>English, French, German and Mexican</b>. Has an <b>Accuracy of 73%</b>. Extracts the features from uploaded audio and classifies into <b>6 different emotions</b> based on a custom trained Neural Network</li>
<br>
<li><b>Model 4</b> - Converts uploaded audio to text using a <b>pretrained Automatic Speech Recognition(ASR) model</b> on <b> HuggingFace transformers</b>. Performs Sentiment Analysis on the <b>returned text using vaderSentiment model</b>. It classifies into <b>3 different classes</b>.
</ol>
    <br>
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Deployment](#deployment)
- [Built Using](#built_using)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

Exploring the field of Sentiment Analysis on a combination of languages and testing model performance. This project directory includes the cleansed dataset as well as the code files to refine and improve the models as well. 

<b>RECOMMENDED: To use the final project, just load up the Google Colab file and follow instructions on how to run the project. </b>

## 🏁 Getting Started <a name = "getting_started" id="getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites

If loading the project locally, use the following command.

```
pip install -r requirements.txt
```

## 🎈 Usage <a name="usage" id="usage"></a>
<b> Follow the instructions on the GOOGLE COLAB file.</b><br>

image.png

On the Gradio Interface, you can find a tab each dedicated to using a different model. You have to click on the upload `audio box` and upload the `audio file`. Then click on the `Submit` button. Wait for a few seconds to find the label of the Sentiment Class displayed in the box below.

## 🚀 Deployment <a name = "deployment" id="deployment"></a>

Visit https://huggingface.co/spaces and create and account there. Upload the code and host your model there to be able to have a 24/7 hosted model, which is accessible anytime.

## ⛏️ Built Using <a name = "built_using" id="built_using"></a>

### Python Libraries used
- [Tensorflow](https://www.tensorflow.org/) - Model Training and building
- [Pandas](https://pandas.pydata.org/) - Dataset building and Data Exploration
- [Matplotlib](https://matplotlib.org/) - Visualizing Sound Waves and plotting Model History in different graphs
- [Librosa](https://github.com/librosa/librosa) - Loading in the audio and extracting audio features
- [HuggingFace Transformers](https://huggingface.co/) - Loaded in the Pretrained ASR model
- [Gradio](https://gradio.app/) - Model deployment on the Web with a proper UI

### Dataset Used
- [RAVDESS](https://zenodo.org/record/1188976#.XrC7a5NKjOR) - English Dataset
- [CaFE](https://zenodo.org/record/1478765) - French Dataset
- [EMO-DB](https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb) - Web Framework
- [MESD](https://data.mendeley.com/datasets/cy34mh68j9/5) - Mexican Dataset

## 🎉 Acknowledgements <a name = "acknowledgement" id="acknowledgement"></a>

- [Dataset References](https://superkogito.github.io/)
- Feature Extraction - AnalyticsVidhya
- [Gradio App Creation](https://gradio.app/real_time_speech_recognition/)

