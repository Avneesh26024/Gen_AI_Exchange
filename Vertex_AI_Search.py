from langchain_google_vertexai import VertexAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")

claim = "The Eiffel Tower is in Berlin."
evidence = "The Eiffel Tower is an iconic landmark in Paris, France."

claim_emb = embedding_model.embed_query(claim)
evidence_emb = embedding_model.embed_query(evidence)

# Cosine similarity
similarity = cosine_similarity([claim_emb], [evidence_emb])[0][0]
# Normalize to 0-1
relevance = (similarity + 1) / 2

print("Relevance:", relevance)
