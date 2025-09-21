import os
import uuid
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from typing import List, Dict, Any
import re  # Added for the example usage


# --- HELPER FUNCTION FOR THE EXAMPLE ---
def chunk_text_by_paragraph(raw_text: str, min_chunk_size: int = 50) -> List[str]:
    """Splits a raw text into meaningful chunks based on paragraphs."""
    paragraphs = re.split(r'\n\s*\n', raw_text)
    return [p.strip() for p in paragraphs if len(p.strip()) >= min_chunk_size]


# --- MODIFIED VECTOR DB FUNCTION ---
def insert_into_vertex_vector_db(
        text_chunks: List[str],
        metadata: Dict[str, Any],  # <-- CHANGED: Now accepts a single dictionary
        project_id: str = "575174467987",
        region: str = "us-central1",
        index_id: str = "3761581011226329088",
        service_account_path: str = None,
        embedding_model_name: str = "text-embedding-004"
):
    """
    Generates embeddings and upserts them into a Vertex AI Vector Search Index.
    This version automatically sorts metadata into the correct filterable format.

    Args:
        text_chunks (List[str]): List of text chunks to insert.
        metadata (Dict[str, Any]): A single dictionary of metadata that will be
            applied to every text chunk.
        project_id (str): GCP project ID.
        region (str): GCP region where the vector index is located.
        index_id (str): The ID of the Vertex AI vector index.
        service_account_path (str): Path to your GCP service account JSON key.
        embedding_model_name (str): The name of the embedding model to use.
    """
    if service_account_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path

    print("--- Initializing Vertex AI client. ---")
    try:
        aiplatform.init(project=project_id, location=region)
        model = TextEmbeddingModel.from_pretrained(embedding_model_name)
        print("Vertex AI client and embedding model initialized successfully.")
    except Exception as e:
        print(f"Error initializing Vertex AI: {e}")
        return

    print(f"--- Preparing to embed {len(text_chunks)} text chunks. ---")
    batch_size = 250
    all_embeddings = []
    try:
        for i in range(0, len(text_chunks), batch_size):
            batch_texts = text_chunks[i:i + batch_size]
            print(f"Embedding batch {i // batch_size + 1}...")
            response = model.get_embeddings(batch_texts)
            all_embeddings.extend([embedding.values for embedding in response])
        print("All text chunks have been embedded successfully.")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return

    datapoints_to_upsert = []

    # --- CHANGED: Create the metadata list inside the function ---
    if not text_chunks:
        print("No text chunks to process.")
        return
    metadata_list = [metadata] * len(text_chunks)

    for i, embedding_vector in enumerate(all_embeddings):
        unique_id = str(uuid.uuid4())
        meta = metadata_list[i]

        current_restricts = []
        current_numeric_restricts = []

        for key, value in meta.items():
            if isinstance(value, str):
                current_restricts.append({"namespace": key, "allow_list": [value]})
            elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                current_restricts.append({"namespace": key, "allow_list": value})
            elif isinstance(value, int):
                current_numeric_restricts.append({"namespace": key, "value_int": value})
            elif isinstance(value, float):
                current_numeric_restricts.append({"namespace": key, "value_float": value})

        datapoint = {
            "datapoint_id": unique_id,
            "feature_vector": embedding_vector,
            "restricts": current_restricts,
            "numeric_restricts": current_numeric_restricts,
            "embedding_metadata": {"chunk_text": text_chunks[i]}
        }
        datapoints_to_upsert.append(datapoint)

    print(f"--- Preparing to upsert {len(datapoints_to_upsert)} vectors into index {index_id}. ---")
    try:
        vector_index = aiplatform.MatchingEngineIndex(index_name=index_id)
        vector_index.upsert_datapoints(datapoints=datapoints_to_upsert)
        print(f"Successfully inserted {len(datapoints_to_upsert)} chunks into Vertex AI Vector DB.")
    except Exception as e:
        print(f"Error upserting data into Vertex AI Vector DB: {e}")
        print("IMPORTANT: This error can occur if your index 'Update method' is set to 'Batch' instead of 'Streaming'.")


# --- CHANGED: Example usage updated for the new function signature ---
if __name__ == "__main__":
    # 1. Define a single source document and its corresponding metadata.
    raw_document = """The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups from the Eurasian Steppe. Several walls were built from as early as the 7th century BC, with selective stretches later joined together by Qin Shi Huang (220–206 BC), the first emperor of China.

The best-known sections of the wall were built by the Ming dynasty (1368–1644). The Great Wall stretches from Dandong in the east to Lop Lake in the west, along an arc that roughly delineates the southern edge of Inner Mongolia.

An archaeological survey found that the entire wall with all of its branches measures out to be 21,196 km (13,171 mi). Today, the Great Wall is generally recognized as one of the most impressive architectural feats in history."""

    single_source_metadata = {
        "source_url": "https://en.wikipedia.org/wiki/Great_Wall_of_China",
        "topic": "History",
        "credibility_score": 0.9,
        "tags": ["architecture", "world_wonder", "china"],
        "year_published": 2023
    }

    # 2. Chunk the single document into multiple text chunks.
    print(f"--- Chunking the source document... ---")
    chunks = chunk_text_by_paragraph(raw_document)
    print(f"Document was split into {len(chunks)} chunks.")

    # 3. Call the function with the list of chunks and the single metadata object.
    print("\n--- Calling the vector database insertion function... ---")
    insert_into_vertex_vector_db(
        text_chunks=chunks,
        metadata=single_source_metadata,  # <-- Pass the single dictionary
        index_id="3761581011226329088"
    )