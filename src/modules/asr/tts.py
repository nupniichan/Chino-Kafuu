from gradio_client import Client

class TTS:
    def __init__(self, server_url="http://127.0.0.1:6969/"):
        self.client = Client(server_url)
        self.model_url = "https://huggingface.co/nuponiichan/Chino-Kafuu/resolve/main/Chino-Kafuu.zip?download=true"
        self.model_path = "logs\\Chino-Kafuu\\Chino-Kafuu.pth"
        self._ensure_model()
    
    def _ensure_model(self):
        """Check if model exists on server, if not download it"""
        try:
            print("Checking if Chino-Kafuu model exists on server...")
            result = self.client.predict(
                pth_path=self.model_path,
                api_name="/run_model_information_script"
            )
            print("Model already exists on server. Skipping download.")
        except Exception as e:
            print("Model not found on server. Downloading Chino-Kafuu model...")
            try:
                result = self.client.predict(
                    model_link=self.model_url,
                    api_name="/run_download_script"
                )
                print("Model downloaded successfully!")
                print(result)
            except Exception as download_error:
                print(f"Error downloading model: {download_error}")
    
    # The setting i already set it to make the output as good as possible
    # You can change the settings if you want to
    def synthesize(
        self,
        text,
        tts_voice="ja-JP-NanamiNeural",
        voice_model="logs\\Chino-Kafuu\\Chino-Kafuu.pth",
        index_file="logs\\Chino-Kafuu\\Chino-Kafuu.index",
        param_1=None,
        param_4=None,
        param_5=None,
        param_6=None,
        param_7=None,
        param_8=None,
        param_14=None,
        param_15=None,
        param_16=None,
        param_17=None,
        param_18=None,
        param_19=None,
        param_20=None,
    ):
        result = self.client.predict(
            terms_accepted=True, # Make sure you read the terms and conditions before using it. I dont have any responsibility for any issues if you violate the terms and conditions.
            param_1=param_1 if param_1 is not None else "", # Input path for text file
            param_2=text, # Text to synthesize
            param_3=tts_voice, # TTS voice default is ja-JP-NanamiNeural
            param_4=param_4 if param_4 is not None else 0, # TTS speed
            param_5=param_5 if param_5 is not None else 2, # Pitch
            param_6=param_6 if param_6 is not None else 0.5, # Search feature ratio
            param_7=param_7 if param_7 is not None else 1, # Volume envelope
            param_8=param_8 if param_8 is not None else 0.5, # Protect voiceless consonants
            param_9="rmvpe", # Pitch extraction algorithm
            param_10="assets\\audios\\tts_output.wav", # Output path for TTS audio
            param_11="assets\\audios\\tts_rvc_output.wav", # Output path for RVC audio
            param_12=voice_model, # Voice model
            param_13=index_file, # Index file
            param_14=param_14 if param_14 is not None else False, # Split audio
            param_15=param_15 if param_15 is not None else False, # Autotune
            param_16=param_16 if param_16 is not None else 1, # Autotune strength ( Autotune must be enabled )
            param_17=param_17 if param_17 is not None else False, # Proposed pitch
            param_18=param_18 if param_18 is not None else 255, # Proposed pitch threshold
            param_19=param_19 if param_19 is not None else True, # Clean audio
            param_20=param_20 if param_20 is not None else 0.05, # Clean strength ( Clean audio must be enabled )
            param_21="WAV", # Export format
            param_22="contentvec", # Embedder model
            param_23=None, # Custom embedder
            param_24=0, # Speaker id
            api_name="/enforce_terms_2"
        )
        return result