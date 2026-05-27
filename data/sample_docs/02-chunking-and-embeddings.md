# Chunking and Embeddings

Documents are chunked before embedding because embedding models and prompts both
have limited context windows, and because retrieval is more precise over small,
focused passages than over whole files. If an entire document were embedded as a
single vector, the signal for a specific fact would be diluted by everything else
in the file. Chunking keeps each vector about one coherent idea.

Chunks usually overlap by a small amount so that a sentence spanning a boundary
is not lost. A typical setup is a chunk of several hundred characters or tokens
with an overlap of ten to twenty percent.

An embedding is a fixed-length numeric vector that represents the meaning of a
piece of text. Texts with similar meaning map to nearby vectors. Embeddings can
come from a hosted API, a local sentence-transformers model, or a lightweight
hashing scheme; the retrieval math is identical as long as the vectors are
normalised.
