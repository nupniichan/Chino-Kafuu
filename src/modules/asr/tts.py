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
    def synthesize(self, text, tts_voice="ja-JP-NanamiNeural", 
                   voice_model="logs\\Chino-Kafuu\\Chino-Kafuu.pth",
                   index_file="logs\\Chino-Kafuu\\Chino-Kafuu.index"):
        result = self.client.predict(
            terms_accepted=True, # Make sure you read the terms and conditions before using it. I dont have any responsibility for any issues if you violate the terms and conditions.
            param_1="", # Input path for text file
            param_2=text, # Text to synthesize
            param_3=tts_voice, # TTS voice default is ja-JP-NanamiNeural
            param_4=0, # TTS speed
            param_5=2, # Pitch
            param_6=0.5, # Search feature ratio
            param_7=1, # Volume envelope
            param_8=0.5, # Protect voiceless consonants
            param_9="rmvpe", # Pitch extraction algorithm
            param_10="assets\\audios\\tts_output.wav", # Output path for TTS audio
            param_11="assets\\audios\\tts_rvc_output.wav", # Output path for RVC audio
            param_12=voice_model, # Voice model
            param_13=index_file, # Index file
            param_14=False, # Split audio
            param_15=False, # Autotune
            param_16=1, # Autotune strength ( Autotune must be enabled )
            param_17=False, # Proposed pitch
            param_18=255, # Proposed pitch threshold
            param_19=True, # Clean audio
            param_20=0.05, # Clean strength ( Clean audio must be enabled )
            param_21="WAV", # Export format
            param_22="contentvec", # Embedder model
            param_23=None, # Custom embedder
            param_24=0, # Speaker id
            api_name="/enforce_terms_2"
        )
        return result