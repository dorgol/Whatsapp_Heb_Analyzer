import streamlit as st
from src.parsing import parse_file
from src.analyzing import main, load_and_prepare_data, plot_messages_over_time, average_message_length_by_user, \
    longest_message, most_busy_day, plot_messages_per_hour, find_ngrams_by_user, find_defining_words

st.title("Hebrew Whatsapp Analyzer")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # Do something with the file
    st.write("File uploaded:", uploaded_file.name)
    chat_history = uploaded_file.read().decode('utf-8')
    parse_file(chat_history)


    df = load_and_prepare_data('parsed_chat.csv')
    st.write(f'Total messages count: ', (len(df)))
    fig = plot_messages_over_time(df)
    st.write(df.head())
    st.plotly_chart(fig)

    avg_lengths = average_message_length_by_user(df)
    st.write("Average Message Length by User:")
    st.write(avg_lengths)

    longest_message_info = longest_message(df)
    st.text_area(f"Longest Message\n: (by {longest_message_info['user']})", longest_message_info['text'])

    busy_day = most_busy_day(df)
    st.write(f"Most Busy Day\n:", busy_day)

    fig = plot_messages_per_hour(df)
    st.plotly_chart(fig)

    n_gram_number = st.number_input('n-gram', min_value=1.0,  max_value=10.0, step=1.0)
    if n_gram_number > 0:
        user_ngrams = find_ngrams_by_user(df, int(n_gram_number))

        st.write(user_ngrams)


    tfidf_number = st.number_input('n-gram max', min_value=1.0,  max_value=10.0, step=1.0)
    user_defining_words = find_defining_words(df, 20, tfidf_number)
    st.write(user_defining_words)




