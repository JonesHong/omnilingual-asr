
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
import torch
import logging
from pathlib import Path

# Import unified pipeline
from omnilingual_asr.streaming import StreamingASRPipeline
from omnilingual_asr.text_utils import clean_asr_output

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ASRServer")

app = FastAPI()

# Configuration
MODEL_CARD = "omniASR_LLM_3B"  # Change to "omniASR_LLM_300M" for LLM model
LANG = "cmn_Hant"  # Traditional Chinese (use "cmn_Hans" for Simplified Chinese)
USE_VAD = True  # Set to True for LLM models
ENABLE_STABILITY_PASS = True # Enable two-pass strategy for better accuracy
MAX_SEGMENT_DURATION_MS = 2000  # Force split after 2 seconds (reduced for lower latency)
MIN_SILENCE_DURATION_MS = 500  # Wait 500ms of silence before finalizing (reduced for faster response)
MIN_SPEECH_DURATION_MS = 250  # Minimum speech duration to process (reduced to catch shorter utterances)
CONTEXT_DURATION_MS = 4000  # Context window for stability pass (4 seconds)

# Initialize Pipeline
logger.info("Initializing ASR Pipeline...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
logger.info(f"Using device: {device}, dtype: {dtype}")
logger.info(f"Model: {MODEL_CARD}, Language: {LANG}, VAD: {USE_VAD}")

# Initialize unified pipeline
pipeline = StreamingASRPipeline(
    model_card=MODEL_CARD,
    device=device,
    dtype=dtype,
    lang=LANG,
    use_vad=USE_VAD,
    enable_stability_pass=ENABLE_STABILITY_PASS,
    min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
    max_segment_duration_ms=MAX_SEGMENT_DURATION_MS,
    context_duration_ms=CONTEXT_DURATION_MS
)

logger.info("ASR Pipeline Ready.")

# HTML Template with Preview/Stable support
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Omnilingual ASR Streaming Demo</title>
        <style>
            body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f7fa; color: #333; }
            .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
            h1 { color: #1a73e8; margin-top: 0; font-weight: 600; }
            .info { background: #e8f0fe; padding: 12px 16px; border-radius: 8px; margin-bottom: 20px; font-size: 14px; color: #1967d2; display: flex; gap: 15px; flex-wrap: wrap; }
            .info strong { color: #174ea6; }
            #status { margin-bottom: 15px; font-weight: 500; color: #5f6368; display: flex; align-items: center; gap: 8px; }
            .status-dot { width: 10px; height: 10px; border-radius: 50%; background: #ccc; display: inline-block; }
            .status-dot.connected { background: #34a853; }
            .status-dot.recording { background: #ea4335; animation: pulse 1.5s infinite; }
            
            #result-container {
                background: #fff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                min-height: 300px;
                max-height: 600px;
                overflow-y: auto;
                padding: 20px;
                font-size: 18px;
                line-height: 1.8;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
            }
            
            .segment { margin-bottom: 8px; }
            .stable { color: #202124; }
            .preview { color: #9aa0a6; font-style: italic; }
            
            .controls { margin-bottom: 20px; display: flex; gap: 10px; }
            button { 
                padding: 10px 24px; 
                font-size: 15px; 
                font-weight: 500;
                border: none; 
                border-radius: 20px; 
                cursor: pointer; 
                transition: all 0.2s;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            #startBtn { background: #34a853; color: white; }
            #startBtn:hover { background: #2d8e46; box-shadow: 0 2px 4px rgba(52,168,83,0.3); }
            #startBtn:disabled { background: #dadce0; color: #fff; cursor: not-allowed; box-shadow: none; }
            
            #stopBtn { background: #ea4335; color: white; }
            #stopBtn:hover { background: #c53929; box-shadow: 0 2px 4px rgba(234,67,53,0.3); }
            #stopBtn:disabled { background: #dadce0; color: #fff; cursor: not-allowed; box-shadow: none; }
            
            #clearBtn { background: #f1f3f4; color: #3c4043; }
            #clearBtn:hover { background: #e8eaed; }

            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(234, 67, 53, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(234, 67, 53, 0); }
                100% { box-shadow: 0 0 0 0 rgba(234, 67, 53, 0); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Omnilingual ASR Streaming</h1>
            <div class="info">
                <span><strong>Model:</strong> """ + MODEL_CARD + """</span>
                <span><strong>Language:</strong> """ + LANG + """</span>
                <span><strong>Mode:</strong> """ + ("VAD + Two-Pass" if USE_VAD else "Low Latency (CTC)") + """</span>
            </div>
            
            <div class="controls">
                <button id="startBtn" onclick="startRecording()">
                    <span>▶ Start</span>
                </button>
                <button id="stopBtn" onclick="stopRecording()" disabled>
                    <span>⏹ Stop</span>
                </button>
                <button id="clearBtn" onclick="clearResult()">
                    <span>🗑 Clear</span>
                </button>
            </div>

            <div id="status">
                <span class="status-dot" id="statusDot"></span>
                <span id="statusText">Ready to connect</span>
            </div>
            
            <div id="result-container">
                <span id="committedText" class="stable"></span>
                <span id="previewText" class="preview"></span>
            </div>
        </div>

        <script>
            let ws;
            let audioContext;
            let processor;
            let input;
            let globalStream;

            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const statusText = document.getElementById('statusText');
            const statusDot = document.getElementById('statusDot');
            const committedSpan = document.getElementById('committedText');
            const previewSpan = document.getElementById('previewText');
            const resultContainer = document.getElementById('result-container');

            function updateStatus(msg, type) {
                statusText.innerText = msg;
                statusDot.className = 'status-dot ' + (type || '');
            }

            function clearResult() {
                committedSpan.innerText = '';
                previewSpan.innerText = '';
            }

            async function startRecording() {
                try {
                    updateStatus("Requesting microphone...", "connecting");
                    globalStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    
                    updateStatus("Connecting to server...", "connecting");
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                    
                    ws.onopen = () => {
                        updateStatus("Recording...", "recording");
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                        initAudioProcessing();
                    };

                    ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        
                        if (data.is_stable) {
                            // Stable text: append to committed area
                            committedSpan.innerText += data.text;
                            // Clear preview since stable text supersedes it
                            previewSpan.innerText = ""; 
                        } else {
                            // Preview text: append to preview area (accumulate)
                            // This shows real-time transcription as it happens
                            previewSpan.innerText += data.text;
                        }
                        
                        // Auto-scroll
                        resultContainer.scrollTop = resultContainer.scrollHeight;
                    };

                    ws.onclose = () => {
                        updateStatus("Disconnected", "");
                        stopRecordingUI();
                    };

                    ws.onerror = (error) => {
                        console.error("WebSocket error:", error);
                        updateStatus("Connection error", "");
                    };

                } catch (err) {
                    console.error("Error:", err);
                    updateStatus("Error: " + err.message, "");
                }
            }

            function stopRecording() {
                if (ws) {
                    ws.close();
                }
                if (globalStream) {
                    globalStream.getTracks().forEach(track => track.stop());
                }
                if (audioContext) {
                    audioContext.close();
                }
                stopRecordingUI();
            }

            function stopRecordingUI() {
                startBtn.disabled = false;
                stopBtn.disabled = true;
                updateStatus("Stopped", "");
            }

            function initAudioProcessing() {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                input = audioContext.createMediaStreamSource(globalStream);
                
                // Use larger buffer size for stability
                processor = audioContext.createScriptProcessor(4096, 1, 1);

                input.connect(processor);
                processor.connect(audioContext.destination);

                processor.onaudioprocess = (e) => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        const inputData = e.inputBuffer.getChannelData(0);
                        // Convert to 16-bit PCM if needed, but float32 is fine for our backend
                        ws.send(inputData.buffer);
                    }
                };
            }
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("New WebSocket connection established")
    
    # Create a new pipeline instance for this session, sharing the model
    session_pipeline = StreamingASRPipeline(
        model_card=MODEL_CARD,
        device=device,
        dtype=dtype,
        lang=LANG,
        use_vad=USE_VAD,
        enable_stability_pass=ENABLE_STABILITY_PASS,
        min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
        max_segment_duration_ms=MAX_SEGMENT_DURATION_MS,
        context_duration_ms=CONTEXT_DURATION_MS,
        base_pipeline=pipeline.base_pipeline
    )
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            
            session_pipeline.add_audio(audio_chunk)
            
            for result in session_pipeline.transcribe_available():
                if result.text:
                    cleaned_text = clean_asr_output(result.text)
                    if cleaned_text:
                        # Send JSON with stability flag
                        response = {
                            "text": cleaned_text + (" " if result.is_stable else ""),
                            "is_stable": result.is_stable
                        }
                        await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        # Finish processing
        for result in session_pipeline.finish():
             if result.text and result.is_stable:
                 cleaned_text = clean_asr_output(result.text)
                 if cleaned_text:
                     response = {
                        "text": cleaned_text + " ",
                        "is_stable": True
                     }
                     # Can't send if disconnected, but we log it
                     logger.info(f"Final text: {cleaned_text}")

    except Exception as e:
        logger.error(f"Error in websocket: {e}")
        import traceback
        traceback.print_exc()
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
