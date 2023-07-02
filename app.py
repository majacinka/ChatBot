import os
import time
import urllib.request
import uuid

import openai
import requests
import streamlit as st


def get_audio(prompt_text):
    tts_model = st.text_input("TTS Model Token here: ")

    # FAKEYOU API endpoints
    TTS_REQUEST_URL = "https://api.fakeyou.com/tts/inference"
    TTS_JOB_STATUS_URL = "https://api.fakeyou.com/tts/job/"
    AUDIO_BASE_URL = "https://storage.googleapis.com/vocodes-public"

    # Replace with your desired TTS model token and text
    TTS_MODEL_TOKEN = tts_model
    TEXT_TO_SPEECH = prompt_text

    # Make a TTS request
    tts_request_payload = {
        "tts_model_token": TTS_MODEL_TOKEN,
        "uuid_idempotency_token": uuid.uuid4().hex,  # Generate a unique id
        "inference_text": TEXT_TO_SPEECH,
    }
    try:
        response = requests.post(TTS_REQUEST_URL, json=tts_request_payload)
        response.raise_for_status()  # This will raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return

    response_json = response.json()

    # Get the job token
    job_token = response_json.get("inference_job_token")

    # Initialize audio_path
    audio_path = None

    # Poll the TTS request status
    while True:
        try:
            response = requests.get(TTS_JOB_STATUS_URL + job_token)
            response.raise_for_status()  # This will raise an exception for HTTP errors
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return

        response_json = response.json()
        job_status = response_json.get("state", {}).get("status")

        if job_status == "complete_success":
            audio_path = response_json.get("state", {}).get(
                "maybe_public_bucket_wav_audio_path"
            )
            break
        elif job_status in ["complete_failure", "dead"]:
            print("Job failed.")
            return

        time.sleep(1)  # Wait for a second before polling again

    # Download the audio file
    if audio_path:
        audio_url = AUDIO_BASE_URL + audio_path
        try:
            data = urllib.request.urlretrieve(audio_url, "output.wav")
        except Exception as e:
            print(f"Failed to download file: {e}")
            return

        return st.audio(data[0])


def main():
    # Get user input for API Key
    api_key = st.text_input("Enter your OpenAI API Key: ", type="password")

    # Get user input for Online resources
    youtube_video = st.text_input("Paste Youtube video here: ")
    web_page = st.text_input("Paste a webpage here: ")
    pdf_link = st.text_input("Paste PDF url here: ")

    # Check if at least one resource is provided
    any_resource = any([youtube_video, web_page, pdf_link])

    if any_resource:
        if api_key:
            # Set OpenAI API Key
            os.environ["OPENAI_API_KEY"] = api_key
            try:
                # Only import App and create an instance if the API key is set
                from embedchain import App

                tesla_bot = App()

                # Embed Online Resources
                tesla_bot.add("youtube_video", youtube_video)
                tesla_bot.add("web_page", web_page)
                tesla_bot("pdf_file", pdf_link)

                st.title("🤖⛓️ Your customizable bot")

                url1 = "https://github.com/embedchain/embedchain"
                url2 = "https://huggingface.co/runwayml/stable-diffusion-v1-5"
                text = f"Nikola Tesla bot knows everything about Nikola Tesla and his amazing achievements ⚡️ The bot is built with multiple youtube, PDF resources as well as Nikola Tesla wikipage 📚 Built thanks to ⛓️[EmbedChain]({url1}) and 🎆[Stable Diffusion]({url2})."

                st.markdown(text)

                # Get user input for the question
                user_query = st.text_input("Enter your question:")

                # Format the prompt

                # Submit button
                if st.button("Submit"):
                    try:
                        # Query and display the result
                        result = tesla_bot.query(user_query)
                        st.write(result)

                        # Generate the image with DallE
                        response = openai.Image.create(
                            prompt=result,  # The text description of the desired image
                            n=1,  # The number of images to generate
                            size="1024x1024",  # The size of the generated image
                        )

                        # Get the URL of the image from the response
                        image_url = response["data"][0]["url"]

                        # Display the image in Streamlit
                        st.image(image_url)

                        audio_player = get_audio(result)
                        if audio_player is not None:
                            st.audio(audio_player)

                    except Exception as e:
                        st.write(f"An error occurred: {e}")

            except Exception as e:
                st.write("Invalid API key")
        else:
            st.error(
                "Please provide at least one resource (Youtube video, webpage or PDF link)."
            )


if __name__ == "__main__":
    main()
