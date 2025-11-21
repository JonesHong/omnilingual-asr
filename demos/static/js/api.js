const API_BASE = '';

class API {
    static async getConfig() {
        const response = await fetch(`${API_BASE}/api/config`);
        return await response.json();
    }

    static async loadModel(modelCard) {
        const formData = new FormData();
        formData.append('model', modelCard);

        const response = await fetch(`${API_BASE}/api/load_model`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to load model');
        }
        return await response.json();
    }

    static async transcribeFile(file, model, language, options = {}, onProgress = null) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', model);
        formData.append('language', language);

        // Add optional parameters
        if (options.chunk_duration) formData.append('chunk_duration', options.chunk_duration);
        if (options.overlap) formData.append('overlap', options.overlap);
        if (options.split_on_silence !== undefined) formData.append('split_on_silence', options.split_on_silence);
        if (options.timestamp_format) formData.append('timestamp_format', options.timestamp_format);
        if (options.time_format) formData.append('time_format', options.time_format);

        const response = await fetch(`${API_BASE}/api/transcribe_file`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Transcription failed');
        }

        if (onProgress) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let fullText = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep the last incomplete line

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);
                        if (data.error) throw new Error(data.error);

                        if (data.type === 'chunk' || data.type === 'final') {
                            // For chunks, we append with a separator (newline or space)
                            // The backend sends formatted chunks
                            const separator = options.timestamp_format !== 'none' ? '\n' : ' ';
                            if (fullText && !fullText.endsWith('\n')) fullText += separator;
                            fullText += data.text;

                            onProgress({
                                text: fullText,
                                currentChunk: data.text,
                                isComplete: false
                            });
                        } else if (data.type === 'complete') {
                            onProgress({
                                text: fullText,
                                isComplete: true,
                                duration: data.duration
                            });
                            return { text: fullText, duration: data.duration };
                        }
                    } catch (e) {
                        console.error("Error parsing stream data:", e);
                    }
                }
            }
            return { text: fullText };
        }

        return await response.json();
    }
}
