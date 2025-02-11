"""
Chapter 8: Applying Machine Learning to Sentiment Analysis
Luke Bray
February 11, 2025
"""

################################################################################
# %% Preparing IMDb movie review data for test processing
# In this section, we prepare the IMDb movie review dataset for further processing.
# The code reads in the data from disk, assigns sentiment labels (1 for positive, 0 for negative),
# and combines reviews from the train and test folders into a single DataFrame.
# Since the pandas DataFrame.append method is deprecated, we use pd.concat to build the DataFrame.
################################################################################

import pyprind  # For displaying progress bars
import pandas as pd  # For data manipulation and DataFrame operations
import os  # For interacting with the operating system (file paths)
import sys  # For system-specific parameters and functions

# Set the base path to the unzipped IMDb movie dataset directory
basepath = '/Users/lukebray/Fall2024/Software_Defined_Radios/MLcode/Applying_Machine_Learning_to_Sentiment_Analysis/aclImdb'

# Define label mapping: 'pos' -> 1, 'neg' -> 0
labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000, stream=sys.stdout)  # Initialize progress bar for 50,000 iterations

# Initialize an empty DataFrame with columns 'review' and 'sentiment'
df = pd.DataFrame(columns=['review', 'sentiment'])

# Loop over the two dataset splits: 'test' and 'train'
for s in ('test', 'train'):
    # For each sentiment folder ('pos' and 'neg')
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)  # Construct path to the reviews
        for file in sorted(os.listdir(path)):  # Loop over all review files in the directory
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()  # Read the content of the review file
            # Create a new DataFrame row with the review text and its corresponding sentiment label
            new_row = pd.DataFrame([[txt, labels[l]]], columns=['review', 'sentiment'])
            # Concatenate the new row with the main DataFrame, ignoring the index to reassign sequentially
            df = pd.concat([df, new_row], ignore_index=True)
            pbar.update()  # Update progress bar

# Check the structure of the DataFrame
print(df.head())

import numpy as np
np.random.seed(0)
# Shuffle the DataFrame rows randomly
df = df.reindex(np.random.permutation(df.index))
# Save the DataFrame to a CSV file for later use
df.to_csv('movie_data.csv', index=False, encoding='utf-8')

# Read the data back from CSV and rename columns if necessary
df = pd.read_csv('movie_data.csv', encoding='utf-8')
df = df.rename(columns={"0": "review", "1": "sentiment"})
df.head(3)


################################################################################
# %% Transforming words in to feature vectors
# This section converts text documents into a matrix of token counts.
# CountVectorizer from scikit-learn is used to generate a bag-of-words representation.
# Optionally, one can change the ngram_range (e.g., to two-grams) by modifying the parameters.
################################################################################

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer (uncomment the two-gram line to use bigrams instead)
count = CountVectorizer()
# count = CountVectorizer(ngram_range=(2,2))  # For two-grams

# Create a sample set of documents
docs = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining, the weather is sweet,'
                 'and one and one is two'])
# Fit the CountVectorizer and transform the sample documents into a bag-of-words matrix
bag = count.fit_transform(docs)

# Print the vocabulary learned by the CountVectorizer
print(count.vocabulary_)
# Print the bag-of-words matrix as a dense array
print(bag.toarray())


################################################################################
# %% Assessing word relevancy via term frequency-inverse document frequency (tf-idf)
# This section transforms the count matrix into a tf-idf representation.
# TfidfTransformer scales the raw frequency counts by term frequency and inverse document frequency.
################################################################################

from sklearn.feature_extraction.text import TfidfTransformer

# Initialize TfidfTransformer with typical parameters
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
# Fit the transformer on the count matrix and convert to an array
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())


################################################################################
# %% Cleaning text data
# This section demonstrates text preprocessing: cleaning HTML tags and non-word characters,
# while preserving emoticons. The preprocessor function is applied to each review in the DataFrame.
################################################################################

# Display the last 50 characters of the first review (for inspection)
df.loc[0, 'review'][-50:]

import re  # For regular expressions

def preprocessor(text):
    # Remove HTML tags
    text = re.sub('<[^>]*>', '', text)
    # Extract emoticons from the text
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Remove non-word characters, convert to lowercase, and append the emoticons at the end
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

# Test the preprocessor function on a sample text
print(preprocessor("</a>This :) is :( a test :-)!"))

# Apply the preprocessor to the 'review' column of the DataFrame
df['review'] = df['review'].apply(preprocessor)


################################################################################
# %% Processing documents into tokens
# This section tokenizes text data. First, a simple whitespace-based tokenizer is defined.
# Then, an enhanced version using the PorterStemmer from NLTK is implemented to perform stemming.
################################################################################

def tokenizer(text):
    # Split the text into tokens based on whitespace
    return text.split()

print(tokenizer('runners like running and thus they run'))

from nltk.stem.porter import PorterStemmer  # Import the Porter stemmer from NLTK

porter = PorterStemmer()

def tokenizer_porter(text):
    # Tokenize the text and apply stemming to each word
    return [porter.stem(word) for word in text.split()]

print(tokenizer_porter('runners like running and thus they run'))


################################################################################
# %% Stop words
# This section downloads and applies a list of English stop words from NLTK.
# It then filters out these stop words from tokenized and stemmed text.
################################################################################

import nltk
nltk.download('stopwords')  # Download the stopwords corpus if not already present

from nltk.corpus import stopwords
stop = stopwords.words('english')  # Get the list of English stop words

# Remove stop words from the tokenized and stemmed example text
print([w for w in tokenizer_porter('a runner likes running and runs a lot') if w not in stop])


################################################################################
# %% Training a logistic regression model for document classification
# This section splits the DataFrame into training and test sets for sentiment analysis.
# It then builds a pipeline that converts text to tf-idf features and trains a logistic
# regression classifier. GridSearchCV is used to find optimal hyperparameters.
################################################################################

# Create training and test subsets using the first 25000 reviews for training,
# and the rest for testing.
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

# Import GridSearchCV for hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer with no pre-processing (since we already preprocessed the text)
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

# Define a small parameter grid with different settings for n-grams, stop words, tokenizer choices,
# and logistic regression hyperparameters.
small_param_grid = [
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [None],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    },
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer],
        'vect__use_idf': [False],
        'vect__norm': [None],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    },
]

# Build a pipeline that vectorizes text using tf-idf and then classifies using logistic regression
lr_tfidf = Pipeline([
    ('vect', tfidf),
    ('clf', LogisticRegression(solver='liblinear'))
])

# Use GridSearchCV to find the best hyperparameters over 5-fold cross-validation
gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)

# Fit GridSearchCV on the training data
gs_lr_tfidf.fit(X_train, y_train)

# %% print params
# Output the best hyperparameter configuration and performance metrics
print(f'Best parameter set: {gs_lr_tfidf.best_params_}')
print(f'CV Accuracy: {gs_lr_tfidf.best_score_:.3f}')

# Retrieve the best estimator and evaluate on the test set
clf = gs_lr_tfidf.best_estimator_
print(f'Test Accuracy: {clf.score(X_test, y_test):.3f}')


################################################################################
# %% Working with bigger data - online algorithms and out-of-core learning
# This section demonstrates out-of-core learning by processing documents in mini-batches.
# A stream of documents is generated from the CSV file, and a HashingVectorizer is used
# to transform text into feature vectors. An SGDClassifier is updated incrementally via partial_fit.
################################################################################

import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')

def tokenizer(text):
    # Remove HTML tags from text
    text = re.sub('<[^>]*>', '', text)
    # Extract emoticons from text
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Remove non-word characters, convert to lowercase, and append emoticons
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    # Split text and filter out stop words
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    # Open the CSV file and yield one document and label at a time
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # Skip header
        for line in csv:
            # Assume label is in the second-to-last character and text is the rest
            text, label = line[:-3], int(line[-2])
            yield text, label

# Test the document stream by printing the first document-label pair
print(next(stream_docs(path='movie_data.csv')))

def get_minibatch(doc_stream, size):
    # Retrieve a mini-batch of documents from the document stream
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

# Initialize a HashingVectorizer for converting text into feature vectors
vect = HashingVectorizer(
    decode_error='ignore',
    n_features=2**21,
    preprocessor=None,
    tokenizer=tokenizer
)

# Initialize an SGDClassifier using logistic regression loss (log_loss)
clf = SGDClassifier(loss='log_loss', random_state=1)
doc_stream = stream_docs(path='movie_data.csv')

import pyprind
pbar = pyprind.ProgBar(45)  # Progress bar for 45 mini-batches
classes = np.array([0, 1])  # Define possible classes

# Process 45 mini-batches, each of size 1000, updating the classifier incrementally
for _ in range(45):
    X_train_batch, y_train_batch = get_minibatch(doc_stream, 1000)
    if not X_train_batch:
        break
    X_train_batch = vect.transform(X_train_batch)
    clf.partial_fit(X_train_batch, y_train_batch, classes=classes)
    pbar.update()

# Get a mini-batch of 5000 documents for testing and evaluate the classifier
X_test_batch, y_test_batch = get_minibatch(doc_stream, 5000)
X_test_batch = vect.transform(X_test_batch)
print(f'Accuracy: {clf.score(X_test_batch, y_test_batch):.3f}')

# Finalize the classifier with the last mini-batch
clf = clf.partial_fit(X_test_batch, y_test_batch)


################################################################################
# %% Topic modeling with latent Dirichlet allocation
# In this section, we use Latent Dirichlet Allocation (LDA) to uncover latent topics
# within the movie reviews. The text is first vectorized using CountVectorizer, and LDA
# is applied to the resulting document-term matrix. The top words for each topic are printed.
################################################################################

import pandas as pd
# Read the movie review data from CSV and rename columns if needed
df = pd.read_csv('movie_data.csv', encoding='utf-8')
df = df.rename(columns={"0": "review", "1": "sentiment"})

from sklearn.feature_extraction.text import CountVectorizer
# Initialize CountVectorizer with English stop words, a maximum document frequency of 0.1,
# and a limit of 5000 features
count = CountVectorizer(stop_words='english', max_df=0.1, max_features=5000)

# Fit the CountVectorizer on the review texts and transform them into a document-term matrix
X = count.fit_transform(df['review'].values)

from sklearn.decomposition import LatentDirichletAllocation
# Initialize LDA with 10 topics, using a fixed random state and batch learning method
lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='batch')
# Transform the document-term matrix into topic distributions
X_topics = lda.fit_transform(X)

# Print the shape of the LDA components (topics x terms)
print(lda.components_.shape)

n_top_words = 5  # Number of top words to display per topic
feature_names = count.get_feature_names_out()  # Get the feature names from CountVectorizer
for topic_idx, topic in enumerate(lda.components_):
    print(f'Topic #{topic_idx + 1}:')
    # Print the top n words for the topic
    print(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

# For demonstration, get the indices of reviews most associated with the 6th topic (index 5)
horror = X_topics[:, 5].argsort()[::-1]

# Print out the first three reviews that are most strongly associated with the selected topic
for iter_idx, movie_idx in enumerate(horror[:3]):
    print(f'\nHorror movie #{(iter_idx + 1)}:')
    print(df['review'][movie_idx][:300], '...')
