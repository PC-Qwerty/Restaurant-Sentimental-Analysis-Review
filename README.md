# Restaurant Sentimental Review Analysis

This repository contains code for analyzing restaurant reviews using Natural Language Processing (NLP) techniques and machine learning models. The goal is to classify restaurant reviews as positive or negative based on the sentiments expressed in the text.

## Packages Used

The analysis code uses the following Python libraries:

- **pandas:** For data manipulation and analysis.
- **numpy:** For numerical operations and array handling.
- **nltk:** For natural language processing tasks such as tokenization and stemming.
- **scikit-learn:** For machine learning algorithms and evaluation metrics.
- **matplotlib:** For data visualization and creating plots.
- **seaborn:** For statistical data visualization.

## Code Overview

The code is written in Python and uses the Jupyter Notebook environment. It consists of the following main sections:

1. **Data Loading and Preprocessing:**
   - The restaurant reviews are loaded from the `Restaurant_Reviews.tsv` file using pandas.
   - Text preprocessing techniques such as removing special characters, converting text to lowercase, tokenization, and stemming are applied to the reviews.
   - Stop words are removed to improve the quality of the text data.

2. **Feature Extraction:**
   - The TfidfVectorizer from scikit-learn is used to convert text data into numerical features.
   - The transformed features are used to train machine learning models.

3. **Model Training:**
   - Random Forest Classifier is used as the machine learning model for this analysis.
   - The model is trained on the preprocessed and vectorized text data.
   - Hyperparameter tuning is performed to find the best set of hyperparameters for the Random Forest Classifier.

4. **Model Evaluation:**
   - The trained model is evaluated using accuracy, precision, and recall scores.
   - A confusion matrix is generated to visualize the performance of the model.

5. **Sample Predictions:**
   - The trained model is used to predict the sentiment of sample restaurant reviews.
   - Example reviews are provided to demonstrate the model's predictions.

## Running the Code

To run the code, make sure you have Python and Jupyter Notebook installed on your system. You can open the `Restaurant Analysis.ipynb` notebook in a Jupyter Notebook environment and execute the cells sequentially.

Note: The notebook contains multiple alternative methods for analysis, including neural network models and BERT-based models. These methods are commented out for clarity. If you wish to explore these methods, uncomment the respective code blocks and ensure you have the required libraries installed.

## Dependencies

Ensure you have the following libraries installed. You can install them using `pip`:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn
```
## A brief logic

1. **Data Loading and Preprocessing:**
   - **Loading Data:** The restaurant reviews are loaded from the `Restaurant_Reviews.tsv` file using pandas.
   - **Text Preprocessing:** Special characters are removed, text is converted to lowercase, and tokenization is performed. Stop words (common words like 'the', 'and', etc.) are removed to enhance the quality of the text data.
   
2. **Feature Extraction:**
   - **Tfidf Vectorization (Active Code):** The TfidfVectorizer from scikit-learn is utilized to convert the preprocessed text data into numerical features. This converts words into numerical values, considering their importance in the document and across documents.
   
   - **Alternative Methods (Commented Code):**
     - **Count Vectorization:** Another approach using CountVectorizer is commented out, which counts the frequency of words in the text.
     - **Neural Network Embeddings:** There's an alternative method using neural networks for generating word embeddings, capturing the semantic meanings of words in dense vectors.

3. **Model Training:**
   - **Random Forest Classifier (Active Code):** The Random Forest Classifier is trained on the preprocessed and vectorized text data. Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputs the mode of the classes (classification) of the individual trees.
   
   - **Alternative Methods (Commented Code):**
     - **Multinomial Naive Bayes:** A probabilistic classifier based on Bayes' theorem with an assumption of independence between features.
     - **BERT Model:** A deep learning model based on Transformers, providing contextual embeddings of words and capturing intricate relationships in the text data.

4. **Model Evaluation:**
   - **Metrics Calculation (Active Code):** Accuracy, precision, and recall scores are calculated to evaluate the performance of the trained model.
   - **Confusion Matrix (Active Code):** A confusion matrix is generated to visualize the model's performance in terms of true positive, true negative, false positive, and false negative predictions.
   
   - **Alternative Methods (Commented Code):**
     - The commented neural network section evaluates accuracy, precision, recall, and plots a confusion matrix using a neural network model.
     - The BERT model section calculates accuracy, precision, recall, and plots a confusion matrix using a pre-trained BERT model.

5. **Sample Predictions:**
   - **Prediction Logic (Active Code):** The trained model is used to predict the sentiment of sample restaurant reviews. The reviews are preprocessed similarly to the training data, and the model predicts whether the sentiment is positive or negative.
   
   - **Alternative Methods (Commented Code):**
     - The commented neural network section predicts sentiment using a neural network model.
     - The BERT model section predicts sentiment using a pre-trained BERT model.

In summary, the code demonstrates different approaches to preprocess text data, convert it into numerical features, train machine learning models (Random Forest and Naive Bayes), and evaluate the models using various metrics. It also includes more advanced methods such as neural networks and BERT models for comparison. Different techniques are provided to give users a comprehensive understanding of text classification methods.

Feel free to reach out if you have any questions or need further assistance!
