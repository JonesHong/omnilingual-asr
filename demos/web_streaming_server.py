
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
import torch
import logging
from pathlib import Path

# Import both pipeline types
from omnilingual_asr.streaming import StreamingASRPipeline
from omnilingual_asr.streaming_vad import StreamingASRPipelineVAD
from omnilingual_asr.text_utils import clean_asr_output

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ASRServer")

app = FastAPI()

# Configuration
MODEL_CARD = "omniASR_LLM_3B"  # Change to "omniASR_LLM_300M" for LLM model
LANG = "cmn_Hant"  # Traditional Chinese (use "cmn_Hans" for Simplified Chinese)
USE_VAD = True  # Set to True for LLM models
MAX_SEGMENT_DURATION_MS = 2000  # Force split after 2 seconds (reduced for lower latency)
MIN_SILENCE_DURATION_MS = 500  # Wait 500ms of silence before finalizing (reduced for faster response)
MIN_SPEECH_DURATION_MS = 250  # Minimum speech duration to process (reduced to catch shorter utterances)

# Initialize Pipeline
logger.info("Initializing ASR Pipeline...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
logger.info(f"Using device: {device}, dtype: {dtype}")
logger.info(f"Model: {MODEL_CARD}, Language: {LANG}, VAD: {USE_VAD}")

# Choose pipeline based on configuration
if USE_VAD or "LLM" in MODEL_CARD:
    logger.info("Using VAD-based pipeline (supports LLM models)")
    pipeline_class = StreamingASRPipelineVAD
    pipeline = pipeline_class(
        model_card=MODEL_CARD,
        device=device,
        dtype=dtype,
        lang=LANG,
        use_vad=True,
        max_segment_duration_ms=MAX_SEGMENT_DURATION_MS
    )
else:
    logger.info("Using Stride-based pipeline (CTC models only)")
    pipeline_class = StreamingASRPipeline
    pipeline = pipeline_class(
        model_card=MODEL_CARD,
        chunk_duration=3.0,
        stride_left=0.5,
        stride_right=0.5,
        device=device,
        dtype=dtype,
        lang=LANG
    )

logger.info("ASR Pipeline Ready.")

# HTML Template
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Omnilingual ASR Streaming Demo</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f0f2f5; }
            .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #1a73e8; }
            .info { background: #e8f0fe; padding: 10px; border-radius: 4px; margin-bottom: 15px; font-size: 14px; }
            #status { margin-bottom: 10px; font-weight: bold; color: #666; }
            #result { 
                white-space: pre-wrap; 
                background: #f8f9fa; 
                padding: 15px; 
                border: 1px solid #ddd; 
                border-radius: 4px; 
                min-height: 200px; 
                font-size: 18px;
                line-height: 1.6;
            }
            button { 
                padding: 10px 20px; 
                font-size: 16px; 
                border: none; 
                border-radius: 4px; 
                cursor: pointer; 
                margin-right: 10px;
                transition: background 0.2s;
            }
            #startBtn { background: #34a853; color: white; }
            #startBtn:hover { background: #2d8e46; }
            #startBtn:disabled { background: #ccc; cursor: not-allowed; }
            #stopBtn { background: #ea4335; color: white; }
            #stopBtn:hover { background: #c53929; }
            #stopBtn:disabled { background: #ccc; cursor: not-allowed; }
            #clearBtn { background: #fbbc04; color: white; }
            #clearBtn:hover { background: #f9ab00; }
            .recording { animation: pulse 1.5s infinite; }
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(234, 67, 53, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(234, 67, 53, 0); }
                100% { box-shadow: 0 0 0 0 rgba(234, 67, 53, 0); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Omnilingual ASR Streaming Demo</h1>
            <div class="info">
                <strong>Model:</strong> """ + MODEL_CARD + """ | 
                <strong>Language:</strong> """ + LANG + """ | 
                <strong>Pipeline:</strong> """ + ("VAD-based (LLM)" if USE_VAD else "Stride-based (CTC)") + """
            </div>
            <div id="status">Ready to connect...</div>
            <div>
                <button id="startBtn" onclick="startRecording()">Start Recording</button>
                <button id="stopBtn" onclick="stopRecording()" disabled>Stop Recording</button>
                <button id="clearBtn" onclick="clearResult()">Clear</button>
            </div>
            <hr/>
            <div id="result"></div>
        </div>

        <script>
            let ws;
            let audioContext;
            let processor;
            let input;
            let globalStream;
            let typingQueue = [];
            let isTyping = false;

            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const statusDiv = document.getElementById('status');
            const resultDiv = document.getElementById('result');

            // Typing animation settings
            const TYPING_SPEED = 30; // milliseconds per character (adjustable)

            function updateStatus(msg) {
                statusDiv.innerText = msg;
            }

            function clearResult() {
                resultDiv.innerText = '';
                typingQueue = [];
                isTyping = false;
            }

            async function typeText(text) {
                isTyping = true;
                for (let char of text) {
                    resultDiv.innerText += char;
                    resultDiv.scrollTop = resultDiv.scrollHeight;
                    await new Promise(resolve => setTimeout(resolve, TYPING_SPEED));
                }
                isTyping = false;
                processQueue();
            }

            function processQueue() {
                if (!isTyping && typingQueue.length > 0) {
                    const nextText = typingQueue.shift();
                    typeText(nextText);
                }
            }

            function addToQueue(text) {
                typingQueue.push(text);
                if (!isTyping) {
                    processQueue();
                }
            }

            async function startRecording() {
                try {
                    updateStatus("Requesting microphone access...");
                    globalStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    
                    updateStatus("Connecting to server...");
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                    
                    ws.onopen = () => {
                        updateStatus("Connected! Recording...");
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                        startBtn.classList.add('recording');
                        initAudioProcessing();
                    };

                    ws.onmessage = (event) => {
                        const text = event.data;
                        // Add to typing queue for smooth animation
                        addToQueue(text);
                    };

                    ws.onclose = () => {
                        updateStatus("Disconnected.");
                        stopRecordingUI();
                    };

                    ws.onerror = (error) => {
                        console.error("WebSocket error:", error);
                        updateStatus("Error connecting to server.");
                    };

                } catch (err) {
                    console.error("Error:", err);
                    updateStatus("Error: " + err.message);
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
                startBtn.classList.remove('recording');
                updateStatus("Stopped.");
            }

            function initAudioProcessing() {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                input = audioContext.createMediaStreamSource(globalStream);
                
                processor = audioContext.createScriptProcessor(4096, 1, 1);

                input.connect(processor);
                processor.connect(audioContext.destination);

                processor.onaudioprocess = (e) => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        const inputData = e.inputBuffer.getChannelData(0);
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
    
    # Reuse global pipeline's model, but create new buffer for this session
    if USE_VAD or "LLM" in MODEL_CARD:
        from omnilingual_asr.streaming_vad import VADSegmenter
        # Create a new segmenter (buffer) for this session
        session_segmenter = VADSegmenter(
            min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
            max_segment_duration_ms=MAX_SEGMENT_DURATION_MS
        )
    else:
        from omnilingual_asr.streaming import StrideAudioBuffer
        # Create a new buffer for this session
        session_segmenter = StrideAudioBuffer(
            chunk_duration=3.0,
            stride_left=0.5,
            stride_right=0.5
        )
    
    # Track last sent text to avoid duplicates
    last_text = ""
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            
            # Add audio to session buffer
            session_segmenter.add_audio(audio_chunk)
            
            # Process using global pipeline with session buffer
            if USE_VAD or "LLM" in MODEL_CARD:
                for segment, timestamp, is_final in session_segmenter.get_speech_segments():
                    # Skip very short segments - LLM models need more context
                    if len(segment) < 8000:  # 0.5 seconds minimum
                        continue
                    text = pipeline._process_segment(segment)
                    if text:
                        # Clean text to remove duplicates (LLM models don't have repetition penalty)
                        cleaned_text = clean_asr_output(text)
                        if cleaned_text and cleaned_text != last_text:
                            # Add space between segments for readability
                            await websocket.send_text(cleaned_text + " ")
                            last_text = cleaned_text
            else:
                for chunk, timestamp in session_segmenter.available_chunks():
                    text = pipeline._process_chunk_optimized(chunk)
                    if text and text != last_text:  # Avoid sending duplicate text
                        await websocket.send_text(text)
                        last_text = text
                    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
        # Flush remaining audio
        if USE_VAD or "LLM" in MODEL_CARD:
            for segment, timestamp, is_final in session_segmenter.finish():
                if len(segment) < 1600:
                    continue
                text = pipeline._process_segment(segment)
                # Can't send if disconnected
        else:
            for chunk, timestamp in session_segmenter.finish():
                text = pipeline._process_chunk_optimized(chunk)
                # Can't send if disconnected
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
