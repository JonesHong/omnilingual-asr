// State
let currentMode = 'file'; // 'file' or 'mic'
let streamClient = null;

// DOM Elements
const elements = {
    tabs: {
        file: document.getElementById('tab-file'),
        mic: document.getElementById('tab-mic')
    },
    panels: {
        file: document.getElementById('panel-file'),
        mic: document.getElementById('panel-mic')
    },
    settings: {
        model: document.getElementById('model-select'),
        language: document.getElementById('language-select')
    },
    file: {
        input: document.getElementById('file-input'),
        uploadArea: document.getElementById('upload-area'),
        transcribeBtn: document.getElementById('transcribe-btn'),
        progressBar: document.getElementById('progress-bar'),
        progressFill: document.getElementById('progress-fill'),
        result: document.getElementById('file-result')
    },
    mic: {
        startBtn: document.getElementById('start-btn'),
        stopBtn: document.getElementById('stop-btn'),
        clearBtn: document.getElementById('clear-btn'),
        statusDot: document.getElementById('status-dot'),
        statusText: document.getElementById('status-text'),
        committedText: document.getElementById('committed-text'),
        previewText: document.getElementById('preview-text'),
        container: document.getElementById('mic-result-container')
    }
};

// Initialization
async function init() {
    try {
        const config = await API.getConfig();

        // Populate Models
        config.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            if (model === config.default_model) option.selected = true;
            elements.settings.model.appendChild(option);
        });

        // Populate Languages
        Object.entries(config.languages).forEach(([name, code]) => {
            const option = document.createElement('option');
            option.value = code;
            option.textContent = name;
            if (code === config.default_language) option.selected = true;
            elements.settings.language.appendChild(option);
        });

        setupEventListeners();
    } catch (err) {
        console.error("Initialization failed:", err);
        alert("Failed to connect to server. Please ensure the backend is running.");
    }
}
// Event Listeners
function setupEventListeners() {
    // Tab Switching
    elements.tabs.file.addEventListener('click', () => switchTab('file'));
    elements.tabs.mic.addEventListener('click', () => switchTab('mic'));

    // Advanced Parameters Toggle
    document.getElementById('toggle-advanced').addEventListener('click', () => {
        const panel = document.getElementById('advanced-panel');
        panel.classList.toggle('hidden');
    });

    // File Upload
    elements.file.uploadArea.addEventListener('click', () => elements.file.input.click());
    elements.file.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.file.uploadArea.style.borderColor = 'var(--primary-color)';
    });
    elements.file.uploadArea.addEventListener('dragleave', () => {
        elements.file.uploadArea.style.borderColor = 'var(--border-color)';
    });
    elements.file.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.file.uploadArea.style.borderColor = 'var(--border-color)';
        if (e.dataTransfer.files.length) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    elements.file.input.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelect(e.target.files[0]);
        }
    });

    elements.file.transcribeBtn.addEventListener('click', transcribeFile);

    // Mic Controls
    elements.mic.startBtn.addEventListener('click', startStreaming);
    elements.mic.stopBtn.addEventListener('click', stopStreaming);
    elements.mic.clearBtn.addEventListener('click', clearMicResult);
}

function switchTab(mode) {
    currentMode = mode;

    // Update Tabs
    elements.tabs.file.classList.toggle('active', mode === 'file');
    elements.tabs.mic.classList.toggle('active', mode === 'mic');

    // Update Panels
    elements.panels.file.classList.toggle('hidden', mode !== 'file');
    elements.panels.mic.classList.toggle('hidden', mode !== 'mic');

    // Update Advanced Parameters visibility
    document.getElementById('file-params').classList.toggle('hidden', mode !== 'file');
    document.getElementById('mic-params').classList.toggle('hidden', mode !== 'mic');

    // Stop streaming if switching away
    if (mode === 'file' && streamClient) {
        stopStreaming();
    }
}

// File Transcription Logic
let selectedFile = null;

function handleFileSelect(file) {
    selectedFile = file;
    elements.file.uploadArea.querySelector('p').textContent = `Selected: ${file.name}`;
    elements.file.transcribeBtn.disabled = false;
}

async function transcribeFile() {
    if (!selectedFile) return;

    const model = elements.settings.model.value;
    const language = elements.settings.language.value;

    // Collect advanced parameters
    const options = {
        timestamp_format: document.getElementById('timestamp-format').value,
        time_format: document.getElementById('time-format').value,
        chunk_duration: document.getElementById('chunk-duration').value,
        overlap: document.getElementById('overlap').value,
        split_on_silence: document.getElementById('split-on-silence').checked
    };

    elements.file.transcribeBtn.disabled = true;
    elements.file.progressBar.style.display = 'block';
    elements.file.progressFill.style.width = '0%';
    elements.file.result.textContent = "Transcribing...";

    try {
        const result = await API.transcribeFile(selectedFile, model, language, options, (progress) => {
            // Update UI with partial results
            let displayText = progress.text;
            if (progress.duration) {
                displayText = `[Duration: ${progress.duration.toFixed(2)}s]\n\n${displayText}`;
            }
            elements.file.result.textContent = displayText;

            // Update progress bar (fake progress for now as we don't know total chunks)
            // But we can show activity
            elements.file.progressFill.style.width = '50%';
            elements.file.progressFill.classList.add('pulse');
        });

        elements.file.progressFill.style.width = '100%';
        elements.file.progressFill.classList.remove('pulse');

    } catch (err) {
        elements.file.result.textContent = `Error: ${err.message}`;
        elements.file.progressFill.style.backgroundColor = 'var(--secondary-color)';
    } finally {
        elements.file.transcribeBtn.disabled = false;
        setTimeout(() => {
            elements.file.progressBar.style.display = 'none';
            elements.file.progressFill.style.width = '0%';
            elements.file.progressFill.style.backgroundColor = 'var(--primary-color)';
        }, 2000);
    }
}

// Streaming Logic
async function startStreaming() {
    const model = elements.settings.model.value;
    const language = elements.settings.language.value;

    // Collect advanced parameters
    const contextDuration = parseInt(document.getElementById('context-duration').value);
    // Note: Other VAD parameters (min_speech, min_silence) would need to be supported by the backend
    // For now we just pass context_duration which is supported

    streamClient = new StreamClient('/ws/transcribe_stream', handleStreamMessage, updateStreamStatus);

    try {
        await streamClient.connect({
            model: model,
            language: language,
            use_vad: true,
            context_duration_ms: contextDuration
        });
        await streamClient.startRecording();

        elements.mic.startBtn.disabled = true;
        elements.mic.stopBtn.disabled = false;
        elements.settings.model.disabled = true; // Lock model while streaming
        elements.settings.language.disabled = true;
    } catch (err) {
        console.error("Streaming failed:", err);
        alert("Failed to start streaming: " + err.message);
    }
}

function stopStreaming() {
    if (streamClient) {
        streamClient.stop();
        streamClient = null;
    }
    elements.mic.startBtn.disabled = false;
    elements.mic.stopBtn.disabled = true;
    elements.settings.model.disabled = false;
    elements.settings.language.disabled = false;
}

function clearMicResult() {
    elements.mic.committedText.textContent = '';
    elements.mic.previewText.textContent = '';
}

function handleStreamMessage(data) {
    if (data.type === 'result') {
        if (data.is_stable) {
            elements.mic.committedText.textContent += data.text;
            elements.mic.previewText.textContent = "";
        } else {
            elements.mic.previewText.textContent += data.text;
        }
        // Auto scroll
        elements.mic.container.scrollTop = elements.mic.container.scrollHeight;
    }
}

function updateStreamStatus(status) {
    elements.mic.statusText.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    elements.mic.statusDot.className = 'dot';

    if (status === 'recording') {
        elements.mic.statusDot.classList.add('recording');
    } else if (status === 'connected') {
        elements.mic.statusDot.classList.add('active');
    }
}

// Start
document.addEventListener('DOMContentLoaded', init);
