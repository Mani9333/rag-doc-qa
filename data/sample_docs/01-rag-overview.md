# Retrieval-Augmented Generation (RAG)

Retrieval-augmented generation is a technique that grounds a language model's
answers in an external corpus of documents. Instead of relying only on what the
model memorised during training, a RAG system first retrieves the most relevant
passages for a question and then asks the model to answer using that retrieved
context. This reduces hallucination and lets the system cite its sources.

A RAG pipeline has two phases. During ingestion, documents are split into chunks,
each chunk is converted into an embedding vector, and the vectors are stored in a
vector index. During querying, the question is embedded with the same model, the
most similar chunks are retrieved, and those chunks are inserted into the prompt
so the model can produce a grounded, citable answer.

The main benefits of RAG are freshness and attribution: you can update the
knowledge base without retraining the model, and every answer can point back to
the exact documents it came from.
