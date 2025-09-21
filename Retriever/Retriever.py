import vertexai
from google.cloud import aiplatform_v1
from vertexai.language_models import TextEmbeddingModel
from typing import List, Dict, Any, Optional


def retrieve_chunks(
        query_text: str,
        top_k: int,
        project_number: str = "575174467987",
        region: str = "us-central1",
        index_endpoint_id: str = "4441976398680162304",
        deployed_index_id: str = "genai_index_1756208565601",
        filters: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves the most similar text chunks from a Vertex AI Vector Search Index,
    with an option to pre-filter based on metadata.

    Args:
        query_text (str): The text to search for.
        top_k (int): The number of results to return.
        project_number (str): Your Google Cloud project number.
        region (str): The region where your index is deployed.
        index_endpoint_id (str): The ID of your index endpoint.
        deployed_index_id (str): The ID of the deployed index.
        filters (Optional[Dict[str, List[str]]]): A dictionary for metadata filtering.
            Example: {"region": ["West India"], "dish_type": ["snack"]}
    """
    print(f"--- Starting retrieval for query: '{query_text}' with filters: {filters} ---")
    try:
        # Step 1: Initialize clients
        vertexai.init(project=project_number, location=region)
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        api_endpoint = "740499490.us-central1-575174467987.vdb.vertexai.goog"
        client_options = {"api_endpoint": api_endpoint}
        vector_search_client = aiplatform_v1.MatchServiceClient(client_options=client_options)

        # Step 2: Generate the embedding for the query text
        response = embedding_model.get_embeddings([query_text])
        query_embedding = response[0].values

        # Step 3: Build the full resource name for the index endpoint
        index_endpoint_name = f"projects/{project_number}/locations/{region}/indexEndpoints/{index_endpoint_id}"

        # Step 4: Build the request object for the vector search
        # --- Build metadata filter restrictions ---
        # Normalize and support multiple filter formats coming from LLM/tool calls.
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Check later whether we need _normalize_filters later or not.
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        def _normalize_filters(filters_input):
            """
            Accepts a variety of filter formats and returns a mapping of
            {metadata_key: [allowed_values...]}. Handles:
            - None
            - dict like {"region": ["North India"], ...}
            - dict with boolean operators: {"and": [{"region": "North India"}, {"dish_type": "snack"}]}
            - protobuf-like objects that implement to_dict() or have __dict__
            - JSON string
            """
            if not filters_input:
                return {}
            import json
            # Try to convert to a plain dict first
            try:
                if isinstance(filters_input, str):
                    filters_obj = json.loads(filters_input)
                elif isinstance(filters_input, dict):
                    filters_obj = filters_input
                else:
                    if hasattr(filters_input, "to_dict"):
                        filters_obj = filters_input.to_dict()
                    elif hasattr(filters_input, "__dict__"):
                        filters_obj = dict(filters_input.__dict__)
                    else:
                        # fallback: try to convert directly
                        filters_obj = dict(filters_input)
            except Exception:
                filters_obj = {}

            normalized = {}
            # If the LLM returned a boolean operator structure like {"and": [...]}
            if isinstance(filters_obj, dict) and any(k.lower() in ("and", "or", "not") for k in filters_obj.keys()):
                for op, items in filters_obj.items():
                    if op.lower() in ("and", "or") and isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                for kk, vv in item.items():
                                    if isinstance(vv, list):
                                        normalized.setdefault(kk, []).extend(vv)
                                    else:
                                        normalized.setdefault(kk, []).append(vv)
                    elif op.lower() == "not":
                        # For simplicity, treat NOT the same as inclusion (could be extended later).
                        if isinstance(items, list):
                            for item in items:
                                if isinstance(item, dict):
                                    for kk, vv in item.items():
                                        if isinstance(vv, list):
                                            normalized.setdefault(kk, []).extend(vv)
                                        else:
                                            normalized.setdefault(kk, []).append(vv)
            else:
                # Flat mapping: make sure every value is a list
                if isinstance(filters_obj, dict):
                    for kk, vv in filters_obj.items():
                        if isinstance(vv, list):
                            normalized.setdefault(kk, []).extend(vv)
                        else:
                            normalized.setdefault(kk, []).append(vv)

            # Ensure all values are strings
            for k in list(normalized.keys()):
                normalized[k] = [str(x) for x in normalized[k] if x is not None]

            return normalized

        normalized_filters = _normalize_filters(filters)
        restricts = []
        for key, allowed_values in normalized_filters.items():
            restricts.append(
                aiplatform_v1.IndexDatapoint.Restriction(
                    namespace=key,
                    allow_list=allowed_values
                )
            )

        query_obj = aiplatform_v1.FindNeighborsRequest.Query(
            datapoint=aiplatform_v1.IndexDatapoint(
                feature_vector=query_embedding,
                restricts=restricts
            ),
            neighbor_count=top_k
        )

        request = aiplatform_v1.FindNeighborsRequest(
            index_endpoint=index_endpoint_name,
            deployed_index_id=deployed_index_id,
            queries=[query_obj],
            return_full_datapoint=True,
        )

        # Step 5: Execute the search request
        response = vector_search_client.find_neighbors(request)
        print("--- Retrieval complete. ---")

        # Step 6: Process the response (MODIFIED AS REQUESTED)
        retrieved_chunks = []
        if response.nearest_neighbors:
            for neighbor in response.nearest_neighbors[0].neighbors:
                datapoint = neighbor.datapoint

                # embedding_metadata may be None depending on the index; coerce to dict
                embedding_metadata = datapoint.embedding_metadata or {}
                # chunk_text stored under a metadata key, fall back to N/A
                chunk_text = embedding_metadata.get("chunk_text") or embedding_metadata.get("text") or "[N/A]"
                try:
                    full_metadata = dict(embedding_metadata)
                except Exception:
                    # If it's not mapping-like, stringify it
                    full_metadata = {"raw": str(embedding_metadata)}

                # Extract filterable restricts from the datapoint (may be None)
                filterable_restricts = {}
                for r in getattr(datapoint, "restricts", []) or []:
                    try:
                        namespace = getattr(r, "namespace", None)
                        allow_list = list(getattr(r, "allow_list", []))
                        if namespace:
                            filterable_restricts[namespace] = allow_list
                    except Exception:
                        # ignore malformed restrict entries
                        continue

                # Extract numeric restricts from the datapoint (may be None)
                numeric_restricts = {}
                for nr in getattr(datapoint, "numeric_restricts", []) or []:
                    try:
                        namespace = getattr(nr, "namespace", None)
                        value = getattr(nr, "value_int", None) or getattr(nr, "value_float", None) or getattr(nr, "value_double", None)
                        if namespace:
                            numeric_restricts[namespace] = value
                    except Exception:
                        continue

                retrieved_chunks.append({
                    "id": getattr(datapoint, "datapoint_id", getattr(datapoint, "id", None)),
                    "text": chunk_text,
                    "distance": round(getattr(neighbor, "distance", 0) or 0, 4),
                    "full_metadata": full_metadata,
                    "filterable_restricts": filterable_restricts,
                    "numeric_restricts": numeric_restricts
                })
        print(f"--- Retrieved {len(retrieved_chunks)} chunks. ---")
        for chunk in retrieved_chunks:
            print(f"--- Retrieved chunk text: {chunk['text']} ---")
        return retrieved_chunks

    except Exception as e:
        print(f"An error occurred during retrieval: {e}")
        return []


# --- This block is for testing the function directly (MODIFIED AS REQUESTED) ---
if __name__ == '__main__':
    # Configuration
    PROJECT_NUMBER = "575174467987"
    REGION = "us-central1"
    INDEX_ENDPOINT_ID = "4441976398680162304"
    DEPLOYED_INDEX_ID = "genai_index_1756208565601"

    # --- TEST 1: Original Semantic Search (No Filters) ---
    print("\n\n<<<<< RUNNING TEST 1: SEMANTIC SEARCH ONLY >>>>>")
    test_query_1 = "The Eiffel Tower is located in Berlin."
    results_1 = retrieve_chunks(
        query_text=test_query_1,
        top_k=2,
    )
    if results_1:
        for result in results_1:
            print(result)
    else:
        print("Could not retrieve any results for Test 1.")

    # --- TEST 2: Filtered Search ---
    print("\n\n<<<<< RUNNING TEST 2: SEMANTIC SEARCH + METADATA FILTER >>>>>")
    test_query_2 = "Tell me about a snack."
    # We are looking for something that is a "snack" AND a "street_food"
    metadata_filters = {
        "doc_type": "verified_claim",
    }

    results_2 = retrieve_chunks(
        query_text=test_query_2,
        top_k=2,
        filters=metadata_filters
    )

    if results_2:
        print("\n==================== FILTERED SEARCH RESULTS ====================")
        for i, result in enumerate(results_2):
            print(f"\n--- Result {i + 1} ---")
            print(f"Distance: {result['distance']}")
            print(f"ID: {result['id']}")
            print(f"Text: \"{result['text']}\"")
            print(f"Full Metadata: {result['full_metadata']}")
            print(f"Filterable Restricts: {result['filterable_restricts']}")
            print(f"Numeric Restricts: {result['numeric_restricts']}")
        print("===============================================================")
    else:
        print("Could not retrieve any results for Test 2.")
