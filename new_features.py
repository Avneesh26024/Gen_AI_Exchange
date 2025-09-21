from google.genai import types
from google import genai

client = genai.Client(
  vertexai=True, location="us-central1"
)
with open(r"C:\Users\Avneesh\Desktop\images_test.jpg", 'rb') as f:
    image_bytes = f.read()

response = client.models.generate_content(
  model='gemini-2.5-flash',
  contents=[
    types.Part.from_bytes(
      data=image_bytes,
      mime_type='image/jpeg',
    ),
    'Caption this image.'
  ]
)

print(response.text)