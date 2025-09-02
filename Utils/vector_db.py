import os
import uuid
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from typing import List, Dict, Any


def insert_into_vertex_vector_db(
        text_chunks: List[str],
        metadata_list: List[Dict[str, Any]],
        project_id: str = "575174467987",
        region: str = "us-central1",
        index_id: str = "3761581011226329088",
        service_account_path: str = None,  # <-- Changed default to None
        embedding_model_name: str = "text-embedding-004"
):
    """
    Generates embeddings and upserts them into a Vertex AI Vector Search Index.
    This version automatically sorts metadata into the correct filterable format.

    Args:
        text_chunks (List[str]): List of text chunks to insert.
        metadata_list (List[Dict[str, Any]]): A list of dictionaries, where each
            dictionary can have any combination of string or numeric metadata.
        project_id (str): GCP project ID.
        region (str): GCP region where the vector index is located.
        index_id (str): The ID of the Vertex AI vector index.
        service_account_path (str): Path to your GCP service account JSON key.
        embedding_model_name (str): The name of the embedding model to use.
    """
    # Authenticate using the service account if provided, else use ADC
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

    # --- NEW: Automated Metadata Processing ---
    datapoints_to_upsert = []
    for i, embedding_vector in enumerate(all_embeddings):
        unique_id = str(uuid.uuid4())
        meta = metadata_list[i]

        current_restricts = []
        current_numeric_restricts = []

        # Iterate through all metadata items and sort them by type
        for key, value in meta.items():
            if isinstance(value, str):
                current_restricts.append({"namespace": key, "allow_list": [value]})
            elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                current_restricts.append({"namespace": key, "allow_list": value})
            elif isinstance(value, int):
                current_numeric_restricts.append({"namespace": key, "value_int": value})
            elif isinstance(value, float):
                current_numeric_restricts.append({"namespace": key, "value_float": value})
            # You can add more types like 'value_double' if needed

        datapoint = {
            "datapoint_id": unique_id,
            "feature_vector": embedding_vector,
            "restricts": current_restricts,
            "numeric_restricts": current_numeric_restricts,
            "embedding_metadata": {"chunk_text": text_chunks[i]}  # <-- ADD THIS LINE
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


# --- Example Usage with Asymmetrical Metadata ---
if __name__ == "__main__":
    # A completely new dataset about Indian cuisine to avoid matching old data
    sample_texts = [
        "Samosas are a popular Indian snack, typically a fried or baked pastry with a savory filling, such as spiced potatoes, onions, and peas.",
        "Butter chicken, or murgh makhani, is a classic Indian curry made with tender chicken in a mildly spiced tomato sauce.",
        "Biryani is a mixed rice dish with its origins among the Muslims of the Indian subcontinent, often reserved for special occasions.",
        "Vada pav is a vegetarian fast food dish native to the state of Maharashtra, consisting of a deep fried potato dumpling placed inside a bread bun."
    ]

    sample_metadata = [
        {"dish_type": "snack", "region": "North India", "spice_level": 3, "tags": ["vegetarian", "fried"]},
        {"dish_type": "main_course", "region": "North India", "spice_level": 2, "tags": ["non-vegetarian", "curry"]},
        {"dish_type": "main_course", "region": "Various", "spice_level": 4, "tags": ["rice", "celebration"]},
        {"dish_type": "snack", "region": "West India", "spice_level": 5, "tags": ["vegetarian", "street_food"]}
    ]

    insert_into_vertex_vector_db(
        text_chunks=sample_texts,
        metadata_list=sample_metadata,
        index_id="3761581011226329088"  # Your index ID
    )
