import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import emoji
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

def setup_nltk():
    """Download required NLTK data"""
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def initialize_processors():
    """Initialize NLTK processors and stopwords"""
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    sentiment_critical = {
        'not', 'no', 'never', 'nothing', 'nobody', 'none', 'nowhere', 'neither',
        'very', 'really', 'quite', 'rather', 'extremely', 'incredibly', 'absolutely',
        'but', 'however', 'although', 'though', 'yet', 'except',
        'too', 'so', 'such', 'more', 'most', 'less', 'least',
        'only', 'just', 'still', 'even', 'again'
    }
    
    negative_contractions = {
        "don't", "won't", "can't", "shouldn't", "wouldn't", "couldn't",
        "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
        "hadn't", "doesn't", "didn't", "won't", "shan't", "mustn't",
        "mightn't", "needn't"
    }
    
    sentiment_critical.update(negative_contractions)
    
    return stop_words, stemmer, lemmatizer, sentiment_critical

def clean_twts(twt, stop_words, lemmatizer, sentiment_critical):
    """Clean and preprocess tweet text"""
    twt = twt.lower()
    twt = re.sub(r"http\S+|www\S+|https\S+", '', twt)  # remove urls
    twt = re.sub(r"@\w+", '', twt)  # remove mentions
    twt = re.sub(r"#", '', twt)  # remove hashtag symbol
    twt = emoji.replace_emoji(twt, replace='')  # remove emojis
    twt = re.sub(r"[^a-zA-Z\s']", '', twt)  # remove punctuation
    twt = re.sub(r"\s+", ' ', twt).strip()  # clean whitespace
   
    tokens = twt.split()
    tokens = [word for word in tokens if (word not in stop_words or word in sentiment_critical) and len(word) > 1]  # remove stopwords and keep sentiment-critical words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # lemmatize
   
    return ' '.join(tokens)

def load_and_preprocess_data(dataset_path):
    """Load and preprocess the twitter dataset"""
    print("Loading dataset...")
    column_names = ['sentiment', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(dataset_path, encoding='latin', delimiter=',', names=column_names)
    print(f"Original dataset shape: {df.shape}")
    
    df = df.drop(['id', 'date', 'flag', 'user'], axis=1)
    df = df.dropna()
    print(f"Dataset shape after cleaning: {df.shape}")
    
    return df

def process_text_data(df):
    """Process text data with cleaning and tokenization"""
    print("Initializing text processors...")
    stop_words, stemmer, lemmatizer, sentiment_critical = initialize_processors()
    
    print("stopwords being removed:")
    print(list(list(stop_words)))  
    
    print("Cleaning tweets...")
    cleaned_twts = df['text'].apply(lambda x: clean_twts(x, stop_words, lemmatizer, sentiment_critical))
    df['cleaned_text'] = cleaned_twts
    
    lengths = [len(text.split()) for text in cleaned_twts]
    print(f"Average length: {np.mean(lengths):.2f}")
    print(f"95th percentile: {np.percentile(lengths, 95):.2f}")
    
    return df, cleaned_twts

def tokenize_and_pad(cleaned_twts):
    """Tokenize and pad sequences"""
    print("Tokenizing text...")
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(cleaned_twts)
    sequences = tokenizer.texts_to_sequences(cleaned_twts)
    padded_sequences = pad_sequences(sequences, maxlen=20, padding='post', truncating='post')
    
    print("Tokenized and padded sequences shape:", padded_sequences.shape)
    print("Sample padded sequence:", padded_sequences[0])
    
    return padded_sequences, tokenizer

def finalize_dataset(df, padded_sequences):
    """Add padded sequences to dataframe and map sentiment labels"""
    df['padded_text'] = list(padded_sequences)
    df['sentiment'] = df['sentiment'].map({4: 1, 0: 0})
    
    print("Final dataset info:")
    print(f"Shape: {df.shape}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    return df

def save_processed_data(df, output_path):
    """Save processed dataset to CSV"""
    print(f"Saving processed data to {output_path}...")
    
    df_save = df.copy()
    df_save['padded_text'] = df_save['padded_text'].apply(lambda x: ','.join(map(str, x)))
    
    df_save.to_csv(output_path, index=False)
    print(f"Processed data saved successfully!")
    print(f"Saved columns: {list(df_save.columns)}")

def main():
    """Main processing function"""
    dataset_path = 'C:/Users/Juls/Desktop/dsml-projects/TwtSentimentAnalysis/data/twt.csv'
    output_path = 'C:/Users/Juls/Desktop/dsml-projects/TwtSentimentAnalysis/data/preprocessed_twt.csv'

    if not os.path.exists(dataset_path):
        print(f"Error: Input file {dataset_path} not found!")
        print("Please ensure the file exists and update the path if necessary.")
        return
    
    setup_nltk()
    
    try:
        df = load_and_preprocess_data(dataset_path)
        
        df, cleaned_twts = process_text_data(df)
        
        padded_sequences, tokenizer = tokenize_and_pad(cleaned_twts)
        
        df = finalize_dataset(df, padded_sequences)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        save_processed_data(df, output_path)
        
        print("\n" + "="*50)
        print("PROCESSING COMPLETE!")
        print(f"Input file: {dataset_path}")
        print(f"Output file: {output_path}")
        print(f"Processed {len(df)} tweets")
        print("="*50)
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        print("Please check your input file and try again.")

if __name__ == "__main__":
    main()