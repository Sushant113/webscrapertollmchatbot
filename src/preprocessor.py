from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    sentences = sent_tokenize(text)
    tokens = [word_tokenize(sentence.lower()) for sentence in sentences]
    stop_words = set(stopwords.words('english'))
    return [[word for word in sentence if word.isalnum() and word not in stop_words] for sentence in tokens]

def chunk_data_with_topic_modeling(data, num_topics=10, chunk_size=5):
    all_text = " ".join([text for _, text in data])
    preprocessed_text = preprocess_text(all_text)
    
    dictionary = corpora.Dictionary(preprocessed_text)
    corpus = [dictionary.doc2bow(text) for text in preprocessed_text]
    
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    
    chunks = []
    for url, text in data:
        sentences = sent_tokenize(text)
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i+chunk_size])
            chunk_bow = dictionary.doc2bow(preprocess_text(chunk)[0])
            chunk_topics = lda_model.get_document_topics(chunk_bow)
            main_topic = max(chunk_topics, key=lambda x: x[1])[0]
            chunks.append((url, chunk, main_topic))
    return chunks
