# Fake News Detection Using NLP: A Study on BERT and LSTM with GloVe Embeddings

This project explores two different machine learning models, LSTM and BERT, for detecting fake news using Natural Language Processing (NLP) techniques. The LSTM model utilizes GloVe embeddings, while the BERT model leverages a pre-trained transformer architecture for classification. Below are the details and steps for both models.

## Project Structure

- **LSTM Model**: Implements a fake news detection model using LSTM with GloVe word embeddings.
- **BERT Model**: Implements a fake news detection model using the BERT transformer model for classification.
> Download Dataset From Kaggle [[Dataset Link]](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## LSTM Model: Fake News Detection Using GloVe Embeddings

### Overview
This model uses an LSTM (Long Short-Term Memory) architecture along with GloVe embeddings to detect fake news. The model is trained on a dataset of news headlines labeled as "Fake" or "True."

### Data Preprocessing
1. **Loading Data**: The dataset consists of two files, one containing true news and the other containing fake news.
2. **Feature Extraction**: 
   - Features such as the number of words, unique words, characters, special characters, punctuation, and stopwords are extracted.
   - The data is cleaned by removing stopwords and unnecessary parts of the text (such as source information).
3. **Text Tokenization and Padding**: 
   - GloVe embeddings are used to convert words into vector representations.
   - Sequences are padded to a fixed length for input into the LSTM model.

### Model Architecture
- **Embedding Layer**: Uses pre-trained GloVe embeddings to convert text into dense vector representations.
- **LSTM Layer**: A Bidirectional LSTM is used to process the sequential data.
- **Dense Layer**: A fully connected layer with a sigmoid activation function classifies the input as fake or true news.

### Results:
- **Precision**: 0.98
- **Recall**: 0.98
- **F1-Score**: 0.98
- **Accuracy**: 98%

---

## BERT Model: Fake News Detection Using BERT Transformer

### Overview
This model uses a pre-trained BERT model to classify news headlines as either "Fake" or "True." The model uses a custom classification layer on top of the BERT encoder.

### Data Preprocessing
1. **Loading Data**: The true and fake news datasets are loaded and merged.
2. **Text Tokenization**: The text is tokenized and encoded using the BERT tokenizer, ensuring that sequences are padded to a fixed length.

### Model Architecture
- **BERT Encoder**: The pre-trained BERT model is used as the encoder.
- **Custom MLP Layer**: After obtaining the pooled output from BERT, a custom multi-layer perceptron (MLP) is added for classification.
- **Softmax Activation**: The final layer uses a softmax activation to output probabilities for each class.

### Results:
- **Precision**: 0.88
- **Recall**: 0.88
- **F1-Score**: 0.88
- **Accuracy**: 88%

---

## Conclusion

The LSTM model outperformed the BERT model in terms of performance, though all parameters of BERT were frozen, and only the weights of the MLP layers were adjusted. Despite being significantly lighter than BERT, the LSTM model demonstrated impressive performance. Therefore, the LSTM architecture proves to be a useful alternative to recent transformer-based models for use cases such as fake news detection.

---
