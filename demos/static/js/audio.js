class AudioHandler {
    constructor(onAudioData) {
        this.onAudioData = onAudioData;
        this.audioContext = null;
        this.stream = null;
        this.processor = null;
        this.input = null;
    }

    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

            this.input = this.audioContext.createMediaStreamSource(this.stream);

            // Use ScriptProcessor for compatibility (AudioWorklet is better but more complex to setup with single file)
            this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);

            this.input.connect(this.processor);
            this.processor.connect(this.audioContext.destination);

            this.processor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                // Clone data because inputBuffer is reused
                const data = new Float32Array(inputData);
                this.onAudioData(data);
            };

            return true;
        } catch (err) {
            console.error("Error accessing microphone:", err);
            throw err;
        }
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }

        if (this.input) {
            this.input.disconnect();
            this.input = null;
        }

        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }
}

class StreamClient {
    constructor(url, onMessage, onStatusChange) {
        this.url = url;
        this.onMessage = onMessage;
        this.onStatusChange = onStatusChange;
        this.ws = null;
        this.audioHandler = null;
    }

    async connect(config) {
        return new Promise((resolve, reject) => {
            this.onStatusChange('connecting');

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}${this.url}`;

            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                // Send configuration
                this.ws.send(JSON.stringify(config));
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.status === 'ready') {
                    this.onStatusChange('connected');
                    resolve();
                } else if (data.status === 'loading_model') {
                    this.onStatusChange('loading');
                } else if (data.status === 'model_loaded') {
                    // Wait for ready
                } else {
                    this.onMessage(data);
                }
            };

            this.ws.onerror = (error) => {
                console.error("WebSocket error:", error);
                this.onStatusChange('error');
                reject(error);
            };

            this.ws.onclose = () => {
                this.onStatusChange('disconnected');
                this.stop();
            };
        });
    }

    async startRecording() {
        this.audioHandler = new AudioHandler((data) => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(data.buffer);
            }
        });

        await this.audioHandler.start();
        this.onStatusChange('recording');
    }

    stop() {
        if (this.audioHandler) {
            this.audioHandler.stop();
            this.audioHandler = null;
        }

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.onStatusChange('disconnected');
    }
}
