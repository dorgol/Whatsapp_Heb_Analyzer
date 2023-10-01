import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import BertModel, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import Counter
import plotly.express as px


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(text, padding=False, truncation=True, return_tensors="pt")
        return tokens


def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'].squeeze() for item in batch], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'].squeeze() for item in batch], batch_first=True)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def load_data(filepath):
    return pd.read_csv(filepath)


def create_dataloader(data, model: str):
    tokenizer = AutoTokenizer.from_pretrained(model)  # same as 'avichr/heBERT' tokenizer
    texts = data['text'].tolist()
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)  # Adjust batch size
    return dataloader


def analyze_sentiment_in_batches(dataloader):
    model = AutoModelForSequenceClassification.from_pretrained("avichr/heBERT_sentiment_analysis")
    model.eval()
    all_labels = []
    all_scores = []
    label_mapping = {0: 'natural', 1: 'positive', 2: 'negative'}
    for batch in dataloader:
        inputs = {key: val.to(model.device) for key, val in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=-1)
        max_scores, max_labels = torch.max(scores, dim=-1)
        all_labels.extend([label_mapping[label.item()] for label in max_labels])
        all_scores.extend(max_scores.cpu().numpy())
    return all_labels, all_scores


def extract_embeddings(dataloader):
    model = BertModel.from_pretrained('onlplab/alephbert-base')
    all_embeddings = []
    for batch in dataloader:
        inputs = {key: val.squeeze(0).to(model.device) for key, val in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.extend(embeddings.cpu().numpy())
    return all_embeddings


def perform_clustering(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)
    return kmeans.labels_


def ner_batch(texts):
    tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT_NER")
    model = AutoModelForTokenClassification.from_pretrained("avichr/heBERT_NER")

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    batch_entities = []
    for i, text in enumerate(texts):
        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
        entities = []
        for j, token in enumerate(tokens):
            label_idx = outputs.logits[i, j].argmax().item()
            label = model.config.id2label[label_idx]
            if label != 'O':
                entities.append((token, label))
        batch_entities.append(entities)
    return batch_entities


def collect_entity_values(entities_list):
    return [entity[0] for entity in entities_list]




def main():
    data = load_data('parsed_chat.csv')
    data = data.iloc[:160, :]
    dataloader = create_dataloader(data, "avichr/heBERT")
    all_labels, all_scores = analyze_sentiment_in_batches(dataloader)
    data['sentiment'] = all_labels
    data['score'] = all_scores
    embeddings = extract_embeddings(dataloader)
    data['topic'] = perform_clustering(embeddings)
    batch_size = 32
    text_batches = [data['text'][i:i + batch_size].tolist() for i in range(0, len(data), batch_size)]

    # Process each batch
    all_entities = []
    for text_batch in text_batches:
        batch_entities = ner_batch(text_batch)
        all_entities.extend(batch_entities)
    data['entities'] = all_entities
    # 1. Determine types of entities
    all_entity_types = [entity[1] for row_entities in data['entities'] for entity in row_entities]
    unique_entity_types = set(all_entity_types)
    print(f"Unique Entity Types: {unique_entity_types}")

    data['entity_values'] = data['entities'].apply(collect_entity_values)

    # Combine all lists into a single list
    all_entity_values = [value for sublist in data['entity_values'] for value in sublist]

    # Get unique entity values
    unique_entity_values = set(all_entity_values)
    print(f"Unique Entity Values: {unique_entity_values}")

    entity_value_counts_per_user = pd.DataFrame(index=data['user'].unique(), columns=list(unique_entity_values)).\
        fillna(0)

    # Populate the DataFrame with counts
    for index, row in data.iterrows():
        user = row['user']
        entity_values = row['entity_values']
        counts = Counter(entity_values)
        for entity_value, count in counts.items():
            entity_value_counts_per_user.loc[user, entity_value] += count

    # Melt the DataFrame for plotting
    long_form_data = entity_value_counts_per_user.reset_index().melt(id_vars='index', var_name='Entity Value',
                                                                     value_name='Count')

    fig = px.bar(
        long_form_data,
        x='Entity Value',
        y='Count',
        color='index',
        title='Frequency of Each Entity Value per User',
        labels={'user': 'User', 'Count': 'Frequency', 'Entity Value': 'Entity Value'},
        height=400,
        width=800
    )

    # Show the plot
    fig.show()



    data.to_csv('analyzed_whatsapp_history.csv', index=False)


if __name__ == "__main__":
    main()
