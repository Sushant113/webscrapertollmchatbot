from pymilvus import connections, Collection
from rank_bm25 import BM25Okapi
import numpy as np
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet', quiet=True)

def query_expansion(query):
    expanded_query = query.split()
    for word in query.split():
        synsets = wordnet.synsets(word)
        for synset in synsets:
            expanded_query.extend(synset.lemma_names())
    
    return ' '.join(set(expanded_query))

def hybrid_retrieval(query, collection_name, encode_bert_func, top_k=10):
    connections.connect("default", host="localhost", port="19530")
    collection = Collection(collection_name)
    collection.load()

    # BM25 retrieval
    corpus = collection.query(output_fields=["text"])
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_k = np.argsort(bm25_scores)[-top_k:]

    # BERT-based retrieval (DPR-like)
    query_embedding = encode_bert_func(query)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search([query_embedding], "embeddings", search_params, limit=top_k)

    # Combine and re-rank results
    combined_results = set(bm25_top_k) | set([hit.id for hit in results[0]])
    reranked_results = []
    for doc_id in combined_results:
        doc = collection.query(expr=f"id == {doc_id}", output_fields=["text", "url", "topic"])[0]
        bm25_score = bm25_scores[doc_id]
        bert_score = next((hit.distance for hit in results[0] if hit.id == doc_id), 0)
        combined_score = 0.4 * bm25_score + 0.6 * (1 - bert_score)  # Adjust weights as needed
        reranked_results.append((doc, combined_score))

    reranked_results.sort(key=lambda x: x[1], reverse=True)
    return reranked_results[:top_k]
