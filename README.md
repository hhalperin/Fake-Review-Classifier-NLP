# Fake-Review-Classifier

### Introduction / Description

Welcome to the 'Fake Review Classifier' project. In the intricate landscape of e-commerce, the credibility of customer reviews is compromised by the growing prevalence of 'fake' reviews. Our solution? A potent blend of Python, machine learning (ML), and natural language processing (NLP) to combat this issue.

This project is centered around the design and implementation of a sophisticated machine learning classifier that effectively distinguishes between authentic and fabricated reviews. By leveraging advanced NLP techniques and ML models, we create an algorithmic solution capable of discerning the subtleties that differentiate genuine feedback from fraudulent entries. It's a step towards restoring the integrity of user-based content in the e-commerce space and beyond."

### Table of Contents

- [Data](#data)
- [Features & Feature Engineering](#features--feature-engineering)
- [Methodology](#methodology)
- [Usage](#usage)
- [Results](#results)

---

### Data

The dataset, titled `fake_reviews_dataset.csv`, contains 40,432 textual reviews that are labeled either as genuine or fake. This data serves as the foundation for training and evaluating the machine learning models. Each review in the dataset is mapped to a binary classification: 

- `0` representing a fake review 
- `1` representing a genuine review

### Features & Feature Engineering

Two primary feature vectors are constructed from each review:

1. **Textual Vector**: This vector is directly tokenized from the review content, capturing the essence and structure of the review.
  
2. **Linguistic Metrics**: These metrics provide insights into the nature of the text in the review.
    - **Subjectivity**: Leveraging the `TextBlob` library, this metric quantifies the balance between opinionated content and factual data within the review.
    - **Polarity**: Also derived using the `TextBlob` library, it measures the sentiment polarity, indicating the sentiment bias of the review. A positive value indicates a positive sentiment, while a negative value indicates the opposite.

---

### Methodology

The project adopts a multi-model approach to ensure a comprehensive and reliable fake review detection mechanism. The following machine learning models are utilized:

1. **Decision Tree Classifier**: A non-parametric supervised learning model. This model creates decision rules inferred directly from the data features, making it suitable for complex classification tasks.

2. **K-Nearest Neighbors (KNN)**: An instance-based learning algorithm. The KNN algorithm classifies new instances based on a majority vote from its 'k' nearest neighbors, considering their distances.

3. **Neural Network (Multilayer Perceptron)**: This is a feedforward artificial neural network that maps sets of input data onto a set of appropriate outputs. It's known for its capacity to model complex relationships in data.

For evaluation, the performance of these models is gauged using standard metrics such as accuracy, precision, recall, F1-score, and insights from the confusion matrix derived from test datasets.

---

### Usage

To utilize the Fake Review Detection Classifier:
1. **Installation and Dependencies**
    -  Install the following dependencies:
        ```
        pip install requirements.txt
        ```
    - Ensure that you also download the required NLTK datasets using:
      ```python
      import nltk
      nltk.download('popular')
      nltk.download("averaged_perceptron_tagger")
      ```
2. **Data Loading**: 
    - Ensure the `fake_reviews_dataset.csv` file is in the appropriate directory or adjust the path in the code accordingly.
    - Use the `load_data()` function to ingest the dataset into the program.

3. **Feature Extraction**: 
    - Process the reviews through the `linguistics_analysis()` function to obtain the subjectivity and polarity metrics for each review.
    - Tokenize the reviews to generate the Textual Vector.

4. **Model Training & Evaluation**:
    - Split the dataset into training and testing sets.
    - Instantiate the desired machine learning model: Decision Tree, KNN, or Neural Network.
    - Train the model on the training dataset.
    - Evaluate the model's performance on the testing dataset using the provided metrics.

Note: Refer to the `fake_review_detection.py` script for detailed code implementation and further guidance.

---

### Results

#### Accuracy:
```
Decision Tree Accuracy: ~0.72

KNN Accuracy: ~0.67

Neural Network Accuracy: ~0.86
```

#### Classification Reports:
Decision Tree Classification Report
```

              precision    recall  f1-score   support

 Fake Review       0.71      0.73      0.72      5077
 Real Review       0.72      0.70      0.71      5031

    accuracy                           0.72     10108
   macro avg       0.72      0.72      0.72     10108
weighted avg       0.72      0.72      0.72     10108
```

KNN Classification Report
```

              precision    recall  f1-score   support

 Fake Review       0.68      0.63      0.66      5060
 Real Review       0.66      0.70      0.68      5048

    accuracy                           0.67     10108
   macro avg       0.67      0.67      0.67     10108
weighted avg       0.67      0.67      0.67     10108
```

Neural Network Classification Report
```

              precision    recall  f1-score   support

 Fake Review       0.86      0.87      0.86      5055
 Real Review       0.87      0.86      0.86      5053

    accuracy                           0.86     10108
   macro avg       0.86      0.86      0.86     10108
weighted avg       0.86      0.86      0.86     10108
```
#### Confusion Matrices

![Confusion Matrices](https://github.com/SpencerSoalt/Fake-Review-Classifier/blob/main/Confusion%20Matrices.png)
