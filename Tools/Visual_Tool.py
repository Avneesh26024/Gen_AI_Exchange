import argparse
from google.cloud import vision


def analyze_image(image_path_or_uri):
    """
    Analyzes an image for various features and returns the results.

    This function detects text, labels, landmarks, logos, and objects
    in a single API call and returns a structured dictionary.

    Args:
        image_path_or_uri (str): The path to a local image file or a public URI.

    Returns:
        dict: A dictionary containing the analysis results.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    results = {
        "texts": [],
        "labels": [],
        "landmarks": [],
        "logos": [],
        "objects": []
    }

    # Load image from local path or URI
    if image_path_or_uri.startswith(("http://", "https://", "gs://")):
        image.source.image_uri = image_path_or_uri
    else:
        with open(image_path_or_uri, "rb") as image_file:
            content = image_file.read()
        image.content = content

    # Define features for the single, batched request
    features = [
        {"type_": vision.Feature.Type.TEXT_DETECTION},
        {"type_": vision.Feature.Type.LABEL_DETECTION},
        {"type_": vision.Feature.Type.LANDMARK_DETECTION},
        {"type_": vision.Feature.Type.LOGO_DETECTION},
        {"type_": vision.Feature.Type.OBJECT_LOCALIZATION},
    ]

    request = vision.AnnotateImageRequest(image=image, features=features)
    response = client.annotate_image(request=request)

    if response.error.message:
        raise Exception(
            f"{response.error.message}\nFor more info: "
            "https://cloud.google.com/apis/design/errors"
        )

    # 1. Process Text annotations
    for text in response.text_annotations:
        vertices = [f"({v.x},{v.y})" for v in text.bounding_poly.vertices]
        results["texts"].append({
            "description": text.description,
            "bounds": ",".join(vertices)
        })

    # 2. Process Label annotations
    for label in response.label_annotations:
        results["labels"].append({
            "description": label.description,
            "score": label.score
        })

    # 3. Process Landmark annotations
    for landmark in response.landmark_annotations:
        locations = []
        for location in landmark.locations:
            lat_lng = location.lat_lng
            locations.append({"latitude": lat_lng.latitude, "longitude": lat_lng.longitude})
        results["landmarks"].append({
            "description": landmark.description,
            "score": landmark.score,
            "locations": locations
        })

    # 4. Process Logo annotations
    for logo in response.logo_annotations:
        results["logos"].append({
            "description": logo.description,
            "score": logo.score
        })

    # 5. Process Object localization annotations
    for obj in response.localized_object_annotations:
        vertices = []
        for v in obj.bounding_poly.normalized_vertices:
            vertices.append({"x": v.x, "y": v.y})
        results["objects"].append({
            "name": obj.name,
            "confidence": obj.score,
            "vertices": vertices
        })

    return results


# --- Main execution block ---
if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Detects features in an image using the Google Cloud Vision API."
    )
    parser.add_argument(
        "image_file", help="The path to the local image file or a public URI."
    )
    args = parser.parse_args()

    try:
        # Call the function to get the analysis data
        analysis_data = analyze_image(args.image_file)

        # --- Now, print the returned data ---

        # Print Text Detection Results
        if analysis_data["texts"]:
            print("--- Text Detection ---")
            # The first annotation is the full text block
            print(f"Full text found:\n'{analysis_data['texts'][0]['description'].strip()}'\n")
        else:
            print("--- No Text Detected ---")

        print("\n" + "=" * 30 + "\n")

        # Print Label Detection Results
        if analysis_data["labels"]:
            print("--- Label Detection ---")
            print("Found the following labels (tags):")
            for label in analysis_data["labels"]:
                print(f"- {label['description']} (score: {label['score']:.2f})")
        else:
            print("--- No Labels Detected ---")

        print("\n" + "=" * 30 + "\n")

        # Print Landmark Detection Results
        if analysis_data["landmarks"]:
            print("--- Landmark Detection ---")
            for landmark in analysis_data["landmarks"]:
                print(f"Landmark: {landmark['description']} (score: {landmark['score']:.2f})")
                for loc in landmark['locations']:
                    print(f"  - Lat/Lng: {loc['latitude']:.4f}, {loc['longitude']:.4f}")
        else:
            print("--- No Landmarks Detected ---")

        print("\n" + "=" * 30 + "\n")

        # Print Logo Detection Results
        if analysis_data["logos"]:
            print("--- Logo Detection ---")
            for logo in analysis_data["logos"]:
                print(f"Logo: {logo['description']} (score: {logo['score']:.2f})")
        else:
            print("--- No Logos Detected ---")

        print("\n" + "=" * 30 + "\n")

        # Print Object Localization Results
        if analysis_data["objects"]:
            print(f"--- Object Localization ({len(analysis_data['objects'])} found) ---")
            for obj in analysis_data["objects"]:
                print(f"Object: {obj['name']} (confidence: {obj['confidence']:.2f})")
        else:
            print("--- No Objects Localized ---")

    except FileNotFoundError:
        print(f"Error: The file '{args.image_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
