import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
import re
import nltk
import sounddevice as sd
import soundfile as sf

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
seed = 47


def clean_and_lemmatize_text(text):
    text = text.lower()  
    text = re.sub(r'\[.*?\]', '', text) 
    text = re.sub(r'\W', ' ', text)  
    text = re.sub(r'\s+', ' ', text) 
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


param_grid = [
    {'classifier': [LogisticRegression(n_jobs=-1, random_state=seed)],
     'classifier__C': [0.1, 1, 10]},
    {'classifier': [RandomForestClassifier(n_jobs=-1, random_state=seed)],
     'classifier__n_estimators': [100, 200]},
    {'classifier': [SVC(random_state=seed)],
     'classifier__C': [0.1, 1, 10, 100],
     'classifier__kernel': ['linear', 'rbf']}
]


def train_models():
    folders = ["fake_news", "job_scams", "phishing", "political_statements", "product_reviews", "sms", "twitter_rumours"]
    folders.reverse()
    for folder in folders:
        play_sound('start.wav')

        print('Starting ', folder)

        train = pd.read_json(f'ds/difraud/{folder}/train.jsonl', lines=True)
        test = pd.read_json(f'ds/difraud/{folder}/test.jsonl', lines=True)
        #
        train['clean_text'] = train['text'].apply(clean_and_lemmatize_text)
        test['clean_text'] = test['text'].apply(clean_and_lemmatize_text)
        #
        print('TFIDF ', folder)
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train = tfidf.fit_transform(train['clean_text']).toarray()
        y_train = train['label'] 
        #
        grid_search = GridSearchCV(estimator=Pipeline([('classifier', LogisticRegression(n_jobs=-1))]), param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print("Best parameters for ", folder, ":", grid_search.best_params_)
        #
        X_test = tfidf.fit_transform(test['clean_text']).toarray()
        y_test = test['label'] 
        #
        y_pred = grid_search.best_estimator_.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        print(f'ROC-AUC Score: {roc_auc:.2f}')
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print('Cross-validation ', folder)
        # Cross-validation
        cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
        print("Cross-validation scores:", cv_scores)
        print("Mean cross-validation score:", cv_scores.mean())

        # Extract the best hyperparameters
        best_params = grid_search.best_params_

        # Define base models with best hyperparameters
        log_clf = LogisticRegression(C=best_params['classifier__C'] if 'classifier__C' in best_params else 1)
        rf_clf = RandomForestClassifier(n_estimators=best_params['classifier__n_estimators'] if 'classifier__n_estimators' in best_params else 100)
        svm_clf = SVC(C=best_params['classifier__C'] if 'classifier__C' in best_params else 1, kernel=best_params['classifier__kernel'] if 'classifier__kernel' in best_params else 'rbf', probability=True)

        # Voting classifier
        voting_clf = VotingClassifier(
            estimators=[('lr', log_clf), ('rf', rf_clf), ('svc', svm_clf)],
            voting='soft',
            n_jobs=-1
        )
        print('Fitting ', folder)
        # Train the voting classifier
        voting_clf.fit(X_train, y_train)

        # Evaluate
        y_pred = voting_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        print(f'ROC-AUC Score: {roc_auc:.2f}')
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        play_sound('finish.wav')



def play_sound(wav_file):
    data, fs = sf.read(wav_file)
    sd.play(data, fs)



if __name__ == '__main__':
    train_models()
