#!/bin/bash
LLM_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
LLM_FILENAME="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# fast-whisper for fatest response
WHISPER_BASE_URL="https://huggingface.co/Systran/faster-whisper-small/resolve/main"
WHISPER_FILES=("config.json" "model.bin" "tokenizer.json" "vocabulary.txt")

RVC_PTH_URL="https://huggingface.co/nuponiichan/Chino-Kafuu/resolve/main/Chino-Kafuu.pth?download=true"
RVC_PTH_FILENAME="Chino-Kafuu.pth"
RVC_INDEX_URL="https://huggingface.co/nuponiichan/Chino-Kafuu/resolve/main/Chino-Kafuu.index?download=true"
RVC_INDEX_FILENAME="Chino-Kafuu.index"

download_file() {
    local filepath=$1
    local url=$2

    if [ -f "$filepath" ]; then
        echo "[SKIP] File exists: $filepath"
    else
        echo "[DOWNLOADING] $filepath ..."
        curl -f -L -o "$filepath" "$url"
                if [ $? -ne 0 ]; then
            echo "[ERROR] Failed to download $filepath"
            rm -f "$filepath"
        else
            echo "[OKE] Download complete."
        fi
    fi
}

echo "Welcome :D"
echo "----------------------------------------------------"

echo "[1/4] Creating Folder Structure...."
mkdir -p models/llm
mkdir -p models/faster-whisper-small
mkdir -p models/rvc

echo ""
echo "[2/4] Processing LLM..."
download_file "models/llm/$LLM_FILENAME" "$LLM_URL"

echo ""
echo "[3/4] Processing Faster-Whisper (Small)..."
for file in "${WHISPER_FILES[@]}"; do
    download_file "models/whisper/$file" "$WHISPER_BASE_URL/$file"
done

echo ""
echo "[4/4] Processing RVC Model..."
download_file "models/rvc/$RVC_FILENAME" "$RVC_PTH_URL"
download_file "models/rvc/$RVC_INDEX_FILENAME" "$RVC_INDEX_URL"

echo ""
echo "----------------------------------------------------"
echo "Done! Check logs above"
echo "----------------------------------------------------"