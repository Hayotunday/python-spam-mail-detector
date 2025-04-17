# Import necessary libraries
import numpy as np  # For numerical operations (though not heavily used here)
import pandas as pd  # For data manipulation and reading CSV/TSV files
import joblib  # For saving and loading the trained model
import matplotlib.pyplot as plt  # For plotting (though not used in this final version)
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier (though SVC is used in the final model)
from sklearn.svm import SVC  # Import Support Vector Classifier (SVC) model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Import metrics for evaluating model performance (only accuracy_score is used here)
from sklearn.pipeline import Pipeline  # Import Pipeline to chain multiple steps (vectorizer and classifier)
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer for converting text data into numerical features

# --- Data Loading with Error Handling ---
try:
    # Attempt to read the dataset from a tab-separated values (TSV) file
    df = pd.read_csv('spam.tsv', sep='\t')
except FileNotFoundError:
    # Handle the case where the file doesn't exist in the current directory
    print("Error: spam.tsv not found. Please ensure the file exists in the same directory as the script.")
    exit()  # Exit the script if the file cannot be found
except UnicodeDecodeError:
    # Handle potential encoding errors if the file is not in the default UTF-8 encoding
    print("Error: Unable to decode the file with default encoding. Trying 'latin-1' encoding.")
    try:
        # Attempt to read the file again using 'latin-1' encoding, common for this dataset
        df = pd.read_csv('spam.tsv', sep='\t', encoding='latin-1')
    except Exception as e:
        # Handle errors if reading with 'latin-1' also fails
        print(f"Error: Failed to read the file even with 'latin-1' encoding. Details: {e}")
        exit() # Exit the script if the file cannot be read
except LookupError as e:
    # Handle other potential encoding lookup errors
    print(f"Error: Encoding issue: {e}")
    print("Please check the file's encoding or try specifying a different encoding like 'utf-8' or 'latin-1'.")
    exit() # Exit the script if there's an encoding issue
except Exception as e:
    # Handle any other unexpected errors during file reading
    print(f"An unexpected error occurred during file reading: {e}")
    exit() # Exit the script if an unexpected error occurs

# --- Data Preparation ---
# Separate the DataFrame into 'ham' (non-spam) and 'spam' messages
hamDf = df[df['label'] == 'ham']
spamDf = df[df['label'] == 'spam']

# Balance the dataset: Randomly sample 'ham' messages to match the number of 'spam' messages.
# This prevents the model from being biased towards the majority class ('ham') in this potentially imbalanced dataset.
hamDf = hamDf.sample(spamDf.shape[0])

# Concatenate the downsampled 'ham' DataFrame and the 'spam' DataFrame into a single DataFrame.
# ignore_index=True resets the index for the new combined DataFrame.
finalDf = pd.concat([hamDf, spamDf], ignore_index=True)

# --- Train-Test Split ---
# Split the data into training and testing sets.
# X contains the features (the 'message' text).
# Y contains the target labels ('ham' or 'spam').
X_train, X_test, Y_train, Y_test = train_test_split(
    finalDf['message'],  # Features (input text)
    finalDf['label'],    # Target labels
    test_size=0.2,       # Hold out 20% of the data for testing
    random_state=0,      # Set a random state for reproducibility of the split
    shuffle=True,        # Shuffle the data before splitting (important for non-ordered data)
    stratify=finalDf['label'] # Ensure the proportion of 'ham' and 'spam' is the same in both train and test sets
)

# --- Model Definition (Pipeline) ---
# Create a machine learning pipeline to streamline the workflow.
# A pipeline chains multiple steps together: data transformation (TF-IDF) followed by model training (SVC).
model = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Step 1: TF-IDF Vectorizer - Converts text messages into numerical TF-IDF features.
                                   # TF-IDF reflects how important a word is to a document in a collection or corpus.
    ('model', SVC(C=1000, gamma='auto')) # Step 2: Support Vector Classifier (SVC) model.
                                        # C=1000: Regularization parameter. Controls the trade-off between achieving a low training error and a low testing error (generalization).
                                        # gamma='auto': Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. 'auto' uses 1 / n_features.
])

# --- Model Training ---
# Train the entire pipeline (TF-IDF transformation and SVC fitting) on the training data.
# The TfidfVectorizer learns the vocabulary and IDF weights from X_train.
# The SVC model learns the decision boundary based on the TF-IDF features of X_train and the labels Y_train.
model.fit(X_train, Y_train)

# --- Prediction ---
# Use the trained pipeline to make predictions on the unseen test data (X_test).
# The pipeline automatically applies the learned TF-IDF transformation to X_test before feeding it to the SVC predictor.
Y_predict = model.predict(X_test)

# --- Evaluation ---
# Evaluate the model's performance by comparing the predicted labels (Y_predict) with the true labels (Y_test).
# Print the accuracy score, which is the proportion of correct predictions.
print("Accuracy: ", accuracy_score(Y_test, Y_predict))

# --- Model Saving ---
# Save the entire trained pipeline (including the fitted TfidfVectorizer and the trained SVC model) to a file.
# This allows the model to be loaded and used later without retraining.
joblib.dump(model, 'mySVCModel1.pkl')
print("Model saved successfully as mySVCModel1.pkl")
