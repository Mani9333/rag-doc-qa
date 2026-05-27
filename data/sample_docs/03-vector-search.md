# Vector Search and Ranking

A vector store ranks results by similarity between the query vector and each
stored chunk vector. The most common similarity measure is cosine similarity,
which compares the angle between two vectors and ignores their magnitude. When
vectors are L2-normalised, cosine similarity is just their dot product, so the
whole search reduces to one matrix-vector multiplication followed by a top-k
selection.

The top-k results are the k chunks with the highest similarity scores. Choosing
k is a tradeoff: too few and the answer may miss relevant context, too many and
the prompt fills with noise that can distract the model.

For small collections an in-memory NumPy store is more than fast enough. For
millions of vectors you would use an approximate nearest-neighbour index such as
FAISS, HNSW, or a managed vector database, which trade a little recall for a large
speed-up.
