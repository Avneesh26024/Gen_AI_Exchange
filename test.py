from google import genai
from google.genai.types import HttpOptions

client = genai.Client(http_options=HttpOptions(api_version="v1"))
chat_session = client.chats.create(model="gemini-2.5-flash")

for chunk in chat_session.send_message_stream("Why is the sky blue?"):
    print(chunk.text, end="")
# Example response:
# The
#  sky appears blue due to a phenomenon called **Rayleigh scattering**. Here's
#  a breakdown of why:
# ...