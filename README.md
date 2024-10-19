# Multi-Class Sentiment Analysis

This project is a multi-emotion classification tool using a pre-trained BERT model. The application is built with Streamlit to provide an interactive web interface for users to analyze the emotions expressed in a given text.

## Features

- **Emotion Prediction**: Predicts multiple emotions from a given text input.
- **Text Preprocessing**: Splits text into sentences and removes full stops for better analysis.
- **Interactive Interface**: User-friendly interface to input text and view results.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Multi-Class-SentimentAnalysis.git
    cd Multi-Class-SentimentAnalysis
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the pre-trained BERT model files in the `./bert_emotion_classifier` directory.

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Streamlit](https://www.streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)