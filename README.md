<div align="center">

# Chino Kafuu AI System
Inspired by Neuro-sama<br><br>
[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange )](https://github.com/nupniichan/Chino-Kafuu)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)]()

</div>

## Overview

Chino Kafuu is a system designed for interactive voice conversations. It combines speech recognition, natural language understanding, memory management, and text-to-speech synthesis to create engaging interactions. The platform can be adapted for various applications including virtual assistants, chatbots, VTuber systems, customer service bots, and other interactive entertainment applications.

## Features

- **Speech-to-Text**: Real-time audio transcription using Faster-Whisper models
- **Dialog Management**: Context-aware conversation handling with LLM integration
- **Memory System**: Multi-layer memory architecture with short-term and long-term storage
- **Text-to-Speech**: Voice synthesis with emotion and action support
- **Voice Activity Detection**: Automatic speech detection and silence handling
- **Auto-Trigger**: Initiates conversations during idle periods
- **RESTful API**: FastAPI-based backend for easy integration
- **Flexible LLM Support**: Works with local models or OpenRouter API using Llama
- **Flexible Caching**: Works with in-memory or Redis Server

## Use Cases

The system is flexible and can be adapted for various applications:

- **Virtual Assistants**: Personal AI assistants with voice interaction
- **Chatbots**: Customer service and support bots with natural conversation flow
- **VTuber Systems**: Interactive virtual characters with personality and emotions
- **Gaming Companions**: AI characters that interact with players naturally
- **Content Creation**: Automated dialogue generation for media production
> And other applications
## Architecture

The system is organized into several core modules that work together to provide seamless voice interaction:

- **ASR (Automatic Speech Recognition)**: Handles speech-to-text conversion and voice activity detection
- **Dialog**: Manages conversation flow, prompt building, and LLM interaction
- **Memory**: Implements short-term conversation buffer and long-term memory compression
- **Audio**: Processes audio capture and playback
- **API**: Provides REST endpoints for system interaction

### System Architecture Diagram

<img width="2384" height="1082" alt="image" src="https://github.com/user-attachments/assets/fa61df48-d86d-41e0-8d37-9d241501438a" />


The diagram above illustrates how the system components interact to process voice input and generate contextual responses. **Keep in mind that it might be changed.**

## Installation

### Prerequisites

- Python >= 3.9
- CUDA-capable GPU (optional, you can use OpenRouter instead if your GPU is limited)
- Redis server (optional, for distributed memory storage)

### External Services

**Applio ( Optional for Voice Conversion)**

Applio is used for RVC (Retrieval-based Voice Conversion) to convert TTS output into character-specific voice. If you want to use voice conversion features:

1. Clone the Applio repository:
   ```bash
   git clone https://github.com/IAHispano/Applio.git
   ```
   Or download the compiled version from [Hugging Face](https://huggingface.co/IAHispano/Applio/tree/main/Compiled)

2. Start the Applio server:
   - Windows: Double-click `run-applio.bat`
   - Linux/Mac: Run `./run-applio.sh`

   The server should be running on `http://127.0.0.1:6969/` by default.

> **Note**: Voice conversion is optional. The system can work without Applio if you don't need Either RVC features or TTS.

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nupniichan/Chino-Kafuu.git
cd ChinoKafuu
```

2. Install dependencies:
   
> It is recommended to use a virtual environment (conda or venv):
```bash
pip install -r requirements.txt
```

3. Download required models:
```bash
# Run the model installer script
scripts/models-installer.bat  # Windows
# or
scripts/models-installer.sh   # Linux/Mac
```

4. Configure environment variables:
   
   Create a `.env` file in the project root with your settings. See `src/setting.py` for all available configuration options.

## Configuration

The system uses environment variables for configuration. Key settings include:

- **LLM Settings**: Model path, context size, temperature, GPU layers
- **Memory Settings**: Short-term buffer size, token limits, compression thresholds
- **Audio Settings**: Sample rate, chunk size, VAD thresholds
- **API Settings**: Host, port, CORS configuration
- **OpenRouter Settings**: API key and model selection (if using cloud LLM)
- **RVC Settings**: Base URL for Applio server (default: `http://127.0.0.1:6969/`)

See `src/setting.py` for all available configuration options.

## Usage

### Starting the API Server

```bash
python api/app.py
# or
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with interactive documentation at `/docs`.

### API Endpoints

- **STT**: `/stt/transcribe` - Transcribe audio files
- **TTS**: `/tts/synthesize` - Generate speech from text
- **Dialog**: `/dialog/chat` - Send messages and get responses
- **Memory**: `/memory/*` - Manage conversation memory
- **System**: `/system/info` - Get system information

## Project Structure

```
ChinoKafuu/
├── api/                 # FastAPI application and routes
├── src/
│   ├── modules/
│   │   ├── asr/         # Speech recognition and VAD
│   │   ├── audio/       # Audio capture, RVC process and playback
│   │   ├── dialog/      # Conversation orchestration
│   │   └── memory/      # Memory management system
│   └── prompts/         # LLM prompt templates
├── data/                # Data storage directory
├── scripts/             # Utility scripts
└── tests/               # Test suite
```

## Memory System

The system uses a two-layer memory architecture:

- **Short-term Memory**: Stores recent conversation messages in a buffer (Redis or in-memory)
- **Long-term Memory**: Compresses conversations into summaries stored in SQLite database

Memory compression happens automatically when the conversation buffer reaches a threshold, preserving important context while managing token limits.

## Development

The codebase follows a modular design with clear separation of concerns:
- Each module handles a specific domain (ASR, dialog, memory)
- Configuration is centralized in `src/setting.py`
- API routes are organized by feature in `api/routes/`

### Running Tests ( this will be removed soon )

```bash
pytest tests/
```

## Requirements

See `requirements.txt` for the complete list of dependencies. Main packages include:

- FastAPI: Web framework
- faster-whisper: Speech recognition
- llama-cpp-python: Local LLM inference
- redis: Memory caching backend
- librosa: Audio processing

## Disclaimer
This is a non-commercial, fan-made project created for personal interest, experimentation, and educational purposes only. It is not affiliated with or endorsed by the original rights holders. All original terms, characters, and intellectual property belong to their respective owners. No copyright infringement is intended.

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
This project was inspired by Neuro-Sama, various open-source projects, and the character Chino Kafuu from Is the Order a Rabbit?.
Special thanks to the open-source community for providing tools, libraries, and ideas that made experimentation and learning possible. This project exists purely out of admiration and creative interest, with no intention of claiming ownership over any original work.

---

Do you dream of bringing Chino to real life too? Let’s make it happen together <3