import pandas as pd
import plotly.graph_objects as go
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from src.stop_words import hebrew_stop_words
import numpy as np


def load_and_prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['time'] = pd.to_datetime(df['time'], format='%H:%M').dt.time
    return df


def count_messages_per_user(df):
    message_counts = df.groupby('user').size().reset_index(name='message_count')
    return message_counts

def plot_messages_over_time(df):
    user_message_counts = df.groupby(['date', 'user']).size().unstack().fillna(0)
    fig = go.Figure()
    for user in user_message_counts.columns:
        fig.add_trace(go.Bar(x=user_message_counts.index, y=user_message_counts[user], name=user))
    fig.update_layout(title="Number of Messages Over Time",
                      xaxis_title="Date",
                      yaxis_title="Number of Messages",
                      barmode='stack')
    # fig.show()
    return fig


def average_message_length_by_user(df):
    avg_lengths = df.groupby('user')['text'].apply(lambda x: x.str.len().mean()).reset_index(name='average_length')
    return avg_lengths


def longest_message(df):
    longest_message_info = df.loc[df['text'].str.len().idxmax()]
    return longest_message_info


def most_busy_day(df):
    busy_day = df.groupby('date').size().idxmax()
    busy_day_df = df[df['date'] == busy_day]
    return busy_day_df


def plot_messages_per_hour(df):
    df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour
    hour_counts = df.groupby('hour').size().reset_index(name='count')
    fig = go.Figure(data=[go.Bar(x=hour_counts['hour'], y=hour_counts['count'])])
    fig.update_layout(title="Distribution of Messages Per Hour",
                      xaxis_title="Hour",
                      yaxis_title="Number of Messages")
    # fig.show()
    return fig


def find_ngrams(text, n):
    # Remove URLs
    text = re.sub(r'(http|https)://[^\s]*', '', text)
    words = re.findall(r'\w+', text.lower())  # Convert text to lower-case and extract words
    filtered_words = [word for word in words if word not in hebrew_stop_words]  # Remove stop words
    ngrams = zip(*[filtered_words[i:] for i in range(n)])  # Create n-grams
    return Counter([' '.join(gram) for gram in ngrams])  # Return n-grams as a Counter object


def find_ngrams_by_user(df, n):
    user_list = []
    ngram_list = []

    for user, messages in df.groupby('user'):
        all_text = ' '.join(messages['text'])
        ngrams_counter = find_ngrams(all_text, n)
        top_ngrams = ngrams_counter.most_common(10)  # Get the top 10 n-grams
        for ngram, _ in top_ngrams:
            user_list.append(user)
            ngram_list.append(ngram)

    result_df = pd.DataFrame({'user': user_list, 'ngram': ngram_list})
    return result_df


def find_defining_words(df, n, ngram_max):  # default to unigrams and bigrams
    ngram_range = (int(ngram_max), int(ngram_max))
    def remove_urls(text):
        return re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove URLs from each message
    df['text'] = df['text'].apply(remove_urls)

    vectorizer = TfidfVectorizer(stop_words=list(hebrew_stop_words), ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    feature_names = vectorizer.get_feature_names_out()

    # Lists to hold the output data
    users = []
    defining_words_list = []

    for user, messages in df.groupby('user'):
        indices = messages.index
        user_tfidf_matrix = tfidf_matrix[indices, :]
        averaged_vector = user_tfidf_matrix.mean(axis=0)
        words_scores = list(zip(feature_names, np.array(averaged_vector).flatten()))
        sorted_words_scores = sorted(words_scores, key=lambda x: -x[1])
        defining_words = [word for word, score in sorted_words_scores[:n] if score > 0]
        for defining_word in defining_words:
            users.append(user)
            defining_words_list.append(defining_word)

    # Create DataFrame
    result_df = pd.DataFrame({'user': users, 'defining_word': defining_words_list})
    return result_df


def analyze_sentiment_distribution(data):
    sentiment_distribution = data.groupby('user')['sentiment'].value_counts(normalize=True).unstack().fillna(0)
    sentiment_distribution.columns.name = None  # Remove the name of the columns axis
    return sentiment_distribution


def main():
    df = load_and_prepare_data('parsed_chat.csv')

    # plot_messages_over_time(df)

    avg_lengths = average_message_length_by_user(df)
    print("Average Message Length by User:")
    print(avg_lengths)

    longest_message_info = longest_message(df)
    print(f"Longest Message: {longest_message_info['text']} (by {longest_message_info['user']})")

    busy_day = most_busy_day(df)
    print(f"Most Busy Day: {busy_day}")

    # plot_messages_per_hour(df)

    user_ngrams = find_ngrams_by_user(df, 4)

    for user, ngrams in user_ngrams.items():
        print(f"Most common 4-grams for {user}: {ngrams}")

    user_defining_words = find_defining_words(df, 10)
    for user, words in user_defining_words.items():
        print(f'Defining words for {user}: {", ".join(words)}')


if __name__ == "__main__":
    main()
