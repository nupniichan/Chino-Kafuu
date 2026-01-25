@echo off
setlocal
title Models Installer For Window

set LLM_URL=https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
set LLM_FILENAME=Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

:: I used fast-whisper for fatest reponse
set WHISPER_BASE_URL=https://huggingface.co/Systran/faster-whisper-small/resolve/main

set RVC_URL=https://huggingface.co/nupniichan/Chino-Kafuu/resolve/main/Chino%%20Kafuu_330e_14190s.pth?download=true
set RVC_FILENAME=Chino_Kafuu.pth
set RVC_INDEX_URL=https://huggingface.co/nupniichan/Chino-Kafuu/resolve/main/added_Chino%%20Kafuu_v2.index?download=true
set RVC_INDEX_FILENAME=added_Chino_Kafuu_v2.index

echo Welcome :D
echo ----------------------------------------------------

echo [1/4] Checking Folder Structure...
if not exist "models\llm" mkdir "models\llm"
if not exist "models\faster-whisper-small" mkdir "models\faster-whisper-small"
if not exist "models\rvc" mkdir "models\rvc"

echo.
echo [2/4] Processing LLM...
call :DownloadFile "models\llm\%LLM_FILENAME%" "%LLM_URL%"

echo.
echo [3/4] Processing Faster-Whisper (Small)...
call :DownloadFile "models\faster-whisper-small\config.json" "%WHISPER_BASE_URL%/config.json"
call :DownloadFile "models\faster-whisper-small\model.bin" "%WHISPER_BASE_URL%/model.bin"
call :DownloadFile "models\faster-whisper-small\tokenizer.json" "%WHISPER_BASE_URL%/tokenizer.json"
call :DownloadFile "models\faster-whisper-small\vocabulary.txt" "%WHISPER_BASE_URL%/vocabulary.txt"


echo.
echo [4/4] Processing RVC Model...
call :DownloadFile "models\rvc\%RVC_FILENAME%" "%RVC_URL%"
call :DownloadFile "models\rvc\%RVC_INDEX_FILENAME%" "%RVC_INDEX_URL%"

echo.
echo ----------------------------------------------------
echo Done! Check the logs above for any errors
echo ----------------------------------------------------
pause
exit /b

:DownloadFile
set "FILE_PATH=%~1"
set "URL=%~2"

if exist "%FILE_PATH%" (
    echo [SKIP] File exists: %FILE_PATH%
) else (
    echo [DOWNLOADING] %FILE_PATH% ...
    :: -f: Fail silently on HTTP errors
    :: -L: Follow redirects
    curl -f -L -o "%FILE_PATH%" "%URL%"
    
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to download: %FILE_PATH%
        echo         Check URL or Internet connection.
        if exist "%FILE_PATH%" del "%FILE_PATH%"
    ) else (
        echo [OKE] Download complete.
    )
)
exit /b