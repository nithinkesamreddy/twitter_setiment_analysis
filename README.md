# Twitter Sentiment Analysis

A scikit-learn project that classifies tweets as **positive** or **negative**. It uses TF-IDF features and compares three classical machine learning models on the Sentiment140 dataset.

## Overview

The notebook covers the full workflow:

1. Load the Sentiment140 CSV.
2. Keep the polarity and text columns.
3. Remove neutral tweets and map labels to `0` (negative) and `1` (positive).
4. Normalize the text.
5. Create a stratified train/test split.
6. Train Bernoulli Naive Bayes, Linear SVM, and Logistic Regression using scikit-learn pipelines.
7. Evaluate each model and select the best one.
8. Predict positive or negative for text entered by the user.

## Dataset

The notebook expects a CSV file named `data.csv` in the project directory.

- Source: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- Rows: 1,600,000 tweets
- Format: CSV without a header row, latin-1 encoded
- Columns used:
  - Column `0`: polarity label
  - Column `5`: tweet text

Label mapping:

| Original label | Meaning  | Mapped label |
| -------------- | -------- | ------------ |
| `0`            | Negative | `0`          |
| `4`            | Positive | `1`          |
| `2`            | Neutral  | Removed      |

The dataset is about 230 MB and is not included in this repository.

## Project Structure

```
twitter/
├── data.csv          # Sentiment140 dataset (not included)
├── twitter.ipynb     # Analysis, training, evaluation, and prediction
├── requirements.txt  # Python dependencies
├── .gitignore        # Files excluded from version control
└── README.md         # Project documentation
```

## Configuration

All project settings are defined in one place near the top of the notebook:

```python
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5_000
NGRAM_RANGE = (1, 2)
DATA_PATH = Path("data.csv")
LABEL_NAMES = {0: "Negative", 1: "Positive"}
```

## Requirements

- Python 3.9+
- pandas
- scikit-learn

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

1. Place `data.csv` in the project directory.
2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `twitter.ipynb`.
4. Run all cells from top to bottom.

For each model, the notebook prints accuracy and a classification report containing precision, recall, and F1-score.

## Model Design

Each classifier is wrapped in a scikit-learn `Pipeline`:

```python
Pipeline([
    ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)),
    ("classifier", classifier),
])
```

This ensures the TF-IDF vocabulary is fitted only on the training data, preventing data leakage. The train/test split is stratified, and a fixed random seed makes the results reproducible.

## Results

Approximate accuracy on the Sentiment140 test split:

| Model                 | Accuracy |
| --------------------- | -------- |
| Bernoulli Naive Bayes | ~0.77    |
| Linear SVM            | ~0.80    |
| Logistic Regression   | ~0.80    |

Linear SVM and Logistic Regression perform best. Exact results may vary slightly depending on library versions and the split.

## Prediction

The final cell trains the best model and accepts user input:

```python
def predict_sentiment(text: str) -> str:
    """Return Positive or Negative for the given text."""
    prediction = best_pipeline.predict([clean_text(text)])[0]
    return LABEL_NAMES[prediction]

while True:
    user_text = input("Enter a tweet (press Enter to stop): ")
    if not user_text.strip():
        break
    print(f"Prediction: {predict_sentiment(user_text)}")
```

Example:

```
Enter a tweet (press Enter to stop): I love this!
Prediction: Positive
```

Press Enter on an empty line to stop.

## Limitations

- Text normalization is limited to lowercasing and whitespace trimming. URLs, mentions, hashtags, punctuation, and numbers remain in the text.
- The TF-IDF vocabulary is limited to 5,000 features.
- Tweets are informal and noisy, which limits achievable accuracy.
- Neutral tweets are removed, so every prediction is either positive or negative.

## Possible Improvements

- Remove URLs, mentions, hashtags, punctuation, and numbers.
- Expand contractions and normalize repeated characters.
- Tune TF-IDF with `sublinear_tf`, `min_df`, and a larger feature set.
- Tune model hyperparameters with cross-validation.
- Try character n-grams or transformer models such as BERT.
- Persist the best pipeline with `joblib` for reuse outside the notebook.
