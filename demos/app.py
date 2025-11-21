import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

# Import pipeline components
from omnilingual_asr.streaming import StreamingASRPipeline
from omnilingual_asr.enhanced_pipeline import EnhancedASRPipeline, TimestampFormat, TimeFormat
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.text_utils import clean_asr_output

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ASRApp")

app = FastAPI(title="Omnilingual ASR Web Demo")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Configuration Constants
AVAILABLE_MODELS = [
    "omniASR_LLM_3B",
    "omniASR_LLM_1B",
    "omniASR_LLM_300M",
    "omniASR_CTC_1B",
    "omniASR_CTC_300M"
]

AVAILABLE_LANGUAGES = {
    "繁體中文 (Traditional Chinese)": "cmn_Hant",
    "簡體中文 (Simplified Chinese)": "cmn_Hans",
    "English": "eng_Latn",
    "日本語 (Japanese)": "jpn_Jpan",
    "한국어 (Korean)": "kor_Hang",
    "Español (Spanish)": "spa_Latn",
    "Français (French)": "fra_Latn",
    "Deutsch (German)": "deu_Latn"
}

class ModelManager:
    def __init__(self):
        self.current_model_card: Optional[str] = None
        self.base_pipeline: Optional[ASRInferencePipeline] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
    def load_model(self, model_card: str):
        if self.current_model_card == model_card and self.base_pipeline is not None:
            return self.base_pipeline
            
        logger.info(f"Loading model: {model_card}...")
        # Clear GPU memory if switching models
        if self.base_pipeline is not None:
            del self.base_pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        try:
            self.base_pipeline = ASRInferencePipeline(
                model_card=model_card,
                device=self.device,
                dtype=self.dtype
            )
            self.current_model_card = model_card
            logger.info(f"Model {model_card} loaded successfully.")
            return self.base_pipeline
        except Exception as e:
            logger.error(f"Failed to load model {model_card}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    def get_pipeline(self):
        if self.base_pipeline is None:
            # Load default model if none loaded
            return self.load_model(AVAILABLE_MODELS[0])
        return self.base_pipeline

# Global Model Manager
model_manager = ModelManager()

@app.get("/")
async def get():
    return HTMLResponse(content=open(static_dir / "index.html", "r", encoding="utf-8").read())

@app.get("/api/config")
async def get_config():
    return {
        "models": AVAILABLE_MODELS,
        "languages": AVAILABLE_LANGUAGES,
        "default_model": AVAILABLE_MODELS[0],
        "default_language": "cmn_Hant"
    }

@app.post("/api/load_model")
async def api_load_model(model: str = Form(...)):
    if model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model card")
    
    model_manager.load_model(model)
    return {"status": "success", "message": f"Model {model} loaded"}

from fastapi.responses import StreamingResponse
import json

@app.post("/api/transcribe_file")
async def transcribe_file(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str = Form(...),
    chunk_duration: float = Form(30.0),
    overlap: float = Form(1.0),
    split_on_silence: bool = Form(False),
    timestamp_format: str = Form("none"), # none, simple, detailed
    time_format: str = Form("auto") # auto, seconds, mm:ss, hh:mm:ss
):
    # Ensure model is loaded
    if model_manager.current_model_card != model:
        model_manager.load_model(model)
        
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    # Parse enums
    ts_format = TimestampFormat.NONE
    if timestamp_format == "simple":
        ts_format = TimestampFormat.SIMPLE
    elif timestamp_format == "detailed":
        ts_format = TimestampFormat.DETAILED
        
    t_format = TimeFormat.AUTO
    if time_format == "seconds":
        t_format = TimeFormat.SECONDS
    elif time_format == "mm:ss":
        t_format = TimeFormat.MMSS
    elif time_format == "hh:mm:ss":
        t_format = TimeFormat.HHMMSS

    async def transcription_generator():
        try:
            # Load audio
            import torchaudio
            waveform, sr = torchaudio.load(tmp_path)
            
            # Resample if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
                sr = 16000
                
            # Check duration
            duration = waveform.shape[1] / sr
            
            # Validate minimum audio length
            if duration < 0.1:
                yield json.dumps({"error": f"Audio file too short ({duration:.2f}s). Minimum duration is 0.1 seconds."}) + "\n"
                return
            
            # Ensure mono audio
            if waveform.dim() > 1 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Create EnhancedASRPipeline instance just for its helper methods
            enhanced_pipeline = EnhancedASRPipeline(
                model_card=model,
                device=model_manager.device,
                dtype=model_manager.dtype,
                base_pipeline=model_manager.get_pipeline()
            )
            
            # Determine effective time format
            effective_time_format = t_format
            if t_format == TimeFormat.AUTO:
                if duration < 60:
                    effective_time_format = TimeFormat.SECONDS
                elif duration < 3600:
                    effective_time_format = TimeFormat.MMSS
                else:
                    effective_time_format = TimeFormat.HHMMSS

            # If audio is short and no forced split, process directly
            if duration <= enhanced_pipeline.MAX_AUDIO_DURATION and not split_on_silence:
                result = enhanced_pipeline.base_pipeline.transcribe(
                    inp=[{"waveform": waveform.squeeze(0), "sample_rate": sr}],
                    lang=[language]
                )
                
                text = result[0]
                if ts_format != TimestampFormat.NONE:
                    text = enhanced_pipeline._format_result(
                        text=text,
                        start_time=0.0,
                        end_time=duration,
                        timestamp_format=ts_format,
                        time_format=effective_time_format
                    )
                else:
                    text = clean_asr_output(text)
                    
                yield json.dumps({
                    "type": "final",
                    "text": text,
                    "duration": duration
                }) + "\n"
                
            else:
                # Long audio or forced split: Split and process chunks
                if split_on_silence:
                    chunks = enhanced_pipeline._split_by_silence(
                        waveform=waveform,
                        sample_rate=sr
                    )
                else:
                    chunks = enhanced_pipeline._split_audio(
                        waveform=waveform,
                        sample_rate=sr,
                        chunk_duration=chunk_duration,
                        overlap=overlap
                    )
                
                full_text_parts = []
                
                for start_time, chunk_waveform in chunks:
                    chunk_dict = {
                        "waveform": chunk_waveform,
                        "sample_rate": sr
                    }
                    
                    result = enhanced_pipeline.base_pipeline.transcribe(
                        inp=[chunk_dict],
                        lang=[language]
                    )
                    
                    chunk_text = result[0]
                    chunk_duration_actual = len(chunk_waveform) / sr
                    end_time = start_time + chunk_duration_actual
                    
                    # Format chunk result
                    formatted_chunk = chunk_text
                    if ts_format != TimestampFormat.NONE:
                        formatted_chunk = enhanced_pipeline._format_result(
                            text=chunk_text,
                            start_time=start_time,
                            end_time=end_time,
                            timestamp_format=ts_format,
                            time_format=effective_time_format
                        )
                    else:
                        formatted_chunk = clean_asr_output(chunk_text)
                    
                    # Yield partial result
                    yield json.dumps({
                        "type": "chunk",
                        "text": formatted_chunk,
                        "start": start_time,
                        "end": end_time
                    }) + "\n"
                    
                    full_text_parts.append(formatted_chunk)

                # Yield completion message
                yield json.dumps({
                    "type": "complete",
                    "duration": duration
                }) + "\n"

        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return StreamingResponse(transcription_generator(), media_type="application/x-ndjson")
        

@app.websocket("/ws/transcribe_stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Initial handshake to get configuration
    try:
        config_data = await websocket.receive_json()
        model = config_data.get("model", AVAILABLE_MODELS[0])
        lang = config_data.get("language", "cmn_Hant")
        use_vad = config_data.get("use_vad", True)
        context_duration_ms = config_data.get("context_duration_ms", 4000)
        
        # Ensure model is loaded
        if model_manager.current_model_card != model:
            await websocket.send_json({"status": "loading_model", "message": f"Loading {model}..."})
            model_manager.load_model(model)
            await websocket.send_json({"status": "model_loaded"})
            
        # Initialize session pipeline
        session_pipeline = StreamingASRPipeline(
            model_card=model,
            device=model_manager.device,
            dtype=model_manager.dtype,
            lang=lang,
            use_vad=use_vad,
            enable_stability_pass=True, # Always enable for best experience
            context_duration_ms=context_duration_ms,
            base_pipeline=model_manager.get_pipeline()
        )
        
        await websocket.send_json({"status": "ready"})
        
        while True:
            data = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            
            # Skip very small chunks that would cause kernel size errors
            # Minimum 512 samples (32ms at 16kHz) required for VAD
            if len(audio_chunk) < 512:
                logger.warning(f"Skipping audio chunk too small: {len(audio_chunk)} samples")
                continue
            
            session_pipeline.add_audio(audio_chunk)
            
            for result in session_pipeline.transcribe_available():
                if result.text:
                    cleaned_text = clean_asr_output(result.text)
                    if cleaned_text:
                        response = {
                            "type": "result",
                            "text": cleaned_text + (" " if result.is_stable else ""),
                            "is_stable": result.is_stable
                        }
                        await websocket.send_json(response)
                        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        if 'session_pipeline' in locals():
            for result in session_pipeline.finish():
                 if result.text and result.is_stable:
                     cleaned_text = clean_asr_output(result.text)
                     if cleaned_text:
                         # Log final text
                         pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)
