import os
import subprocess
from google.cloud import storage
from google.cloud import speech_v1p1beta1 as speech
from langchain_core.tools import tool

# --- Configuration ---
# IMPORTANT: Update this path to the location of your ffmpeg.exe
# Or, ensure ffmpeg is in your system's PATH environment variable.
FFMPEG_PATH = r"C:\Users\Avneesh\Downloads\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"


def preprocess_audio(file_path: str, target_rate: int = 16000) -> str:
    """
    Resamples an audio file to a target sample rate (default 16kHz) in WAV format using FFmpeg.
    This is a crucial checkpoint before transcription for compatibility and accuracy.

    Args:
        file_path (str): Path to the input audio file.
        target_rate (int): Target sample rate (default: 16000).

    Returns:
        str: Path to the preprocessed audio file, or an empty string on failure.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] Input file not found: {file_path}")
        raise FileNotFoundError(f"Input file does not exist: {file_path}")

    file_dir, file_name = os.path.split(file_path)
    file_root, _ = os.path.splitext(file_name)

    # Always output a WAV file in the same folder with a descriptive suffix.
    output_file = os.path.join(file_dir, f"{file_root}_{target_rate}Hz.wav")

    try:
        # FFmpeg command to convert and resample the audio
        # -y: Overwrite output file if it exists
        # -i: Input file
        # -ar: Audio sample rate
        # -ac: Audio channels (1 for mono)
        command = [
            FFMPEG_PATH, "-y", "-i", file_path,
            "-ar", str(target_rate), "-ac", "1", output_file
        ]

        # Using subprocess.run to execute the command
        process = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"[CHECKPOINT] Preprocessed audio: {file_path} -> {output_file}")
        return output_file
    except FileNotFoundError:
        print(f"[ERROR] FFmpeg executable not found at: {FFMPEG_PATH}")
        print("Please ensure FFmpeg is installed and the FFMPEG_PATH is correct.")
        return ""
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg failed with exit code {e.returncode}.")
        print(f"FFmpeg stderr: {e.stderr}")
        return ""


def upload_to_gcs(local_file: str, bucket_name: str, blob_name: str) -> str:
    """
    Uploads a local file to Google Cloud Storage.

    Args:
        local_file (str): Path to the local file to upload.
        bucket_name (str): The name of the GCS bucket.
        blob_name (str): The desired name of the object in GCS.

    Returns:
        str: The gs:// URI of the uploaded file.
    """
    print(f"[CHECKPOINT] Uploading {local_file} to gs://{bucket_name}/{blob_name}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_file)
    print(f"[CHECKPOINT] Upload complete.")
    return f"gs://{bucket_name}/{blob_name}"


def gcs_transcribe(gcs_uri: str, language_code: str = "en-US") -> str:
    """
    Performs long-running speech recognition on a file in GCS.

    Args:
        gcs_uri (str): The gs:// URI of the audio file.
        language_code (str): The language code for transcription.

    Returns:
        str: The transcribed text, or an empty string on failure.
    """
    client = speech.SpeechClient()

    # Since preprocessing standardizes to WAV, we can rely on LINEAR16 encoding.
    # The sample rate is known from our preprocessing step.
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        enable_automatic_punctuation=True,
    )

    audio = speech.RecognitionAudio(uri=gcs_uri)

    try:
        print(f"[CHECKPOINT] Sending transcription request for {gcs_uri}...")
        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=300)  # 5-minute timeout

        transcript = " ".join([r.alternatives[0].transcript for r in response.results])
        print("[CHECKPOINT] Transcription successful.")
        return transcript
    except Exception as e:
        print(f"[ERROR] Exception from Speech-to-Text API: {e}")
        return ""


@tool
def speech_to_text(file_path: str, bucket_name: str = "genai_bucket_26", language_code: str = "en-US") -> str:
    """
    Converts a local audio file to text using Google Cloud Speech-to-Text.
    The process involves preprocessing the audio, uploading to GCS, and transcribing.

    Args:
        file_path (str): Path to the local audio file (e.g., ./audio.mp3).
        bucket_name (str): GCS bucket where the file will be uploaded.
        language_code (str): Language code for transcription (e.g., "en-US").
    """
    # --- STEP 1: Preprocess the audio file ---
    print("\n--- Starting Transcription Pipeline ---")
    preprocessed_file = preprocess_audio(file_path)
    if not preprocessed_file:
        return "Failed to preprocess audio file. Aborting."

    # --- STEP 2: Upload the preprocessed file to GCS ---
    folder_name = "hf-agent-audio"  # A pseudo-folder in your GCS bucket
    blob_name = f"{folder_name}/{os.path.basename(preprocessed_file)}"
    gcs_uri = upload_to_gcs(preprocessed_file, bucket_name, blob_name)
    print(f"[DEBUG] GCS URI: {gcs_uri}")

    # --- STEP 3: Transcribe the file from GCS ---
    result = gcs_transcribe(gcs_uri, language_code)
    print(f"[DEBUG] gcs_transcribe result: {result}")

    # --- STEP 4: Clean up the local preprocessed file ---
    try:
        os.remove(preprocessed_file)
        print(f"[CHECKPOINT] Cleaned up temporary file: {preprocessed_file}")
    except OSError as e:
        print(f"[WARNING] Could not remove temporary file {preprocessed_file}: {e}")

    print("--- Transcription Pipeline Finished ---")
    return result


# ==============================================================================
# Example of how to run the tool directly for testing
# ==============================================================================
if __name__ == "__main__":
    # --- Configuration for the test run ---
    # 👇 CHANGE THIS to a valid local audio file path on your system.
    test_file = r"C:\Users\Avneesh\Downloads\Monologue_16k.wav"

    # 👇 Your GCS bucket name (must already exist in your GCP project).
    test_bucket = "genai_bucket_26"

    # --- Execute the tool ---
    try:
        # To run the LangChain tool, we use the .invoke() method
        # and pass the arguments as a dictionary.
        transcript = speech_to_text.invoke({
            "file_path": test_file,
            "bucket_name": test_bucket,
            "language_code": "en-US"
        })

        print("\n✅ Transcription Result:")
        print(transcript)

        if not transcript:
            print("\n⚠️ No transcript returned. Check the logs above for errors.")
            print("Possible causes:")
            print("- The audio file might be silent or contain no clear speech.")
            print("- GCS permissions might be incorrect for the service account.")
            print("- The Google Cloud Speech-to-Text API may not be enabled for your project.")

    except FileNotFoundError:
        print(f"❌ Error: The input file was not found. Please check the path: {test_file}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

