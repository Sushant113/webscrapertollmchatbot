from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def answer_question(query, context):
    result = qa_pipeline(question=query, context=context)
    return result['answer']
