/**
 * Voice Studio - Interactive voice preset designer
 * Dark theme edition with polished UI
 */

class VoiceStudio {
    constructor() {
        this.ws = null;
        this.sessionId = null;
        this.audioContext = null;
        this.currentAudio = null;
        this.previewAudioData = null;
        this.isPlaying = false;
        this.playbackStartTime = 0;
        this.playbackOffset = 0;

        // A/B Comparison
        this.abSlots = {
            a: { params: null, audio: null, name: null, voice: null },
            b: { params: null, audio: null, name: null, voice: null }
        };

        // Tags for save modal
        this.currentTags = [];

        // Current parameters
        this.params = {
            formant_ratio: 1.0,
            breath_intensity: 0.15,
            jitter_percent: 0.5,
            pitch_drift_cents: 8.0,
            timing_variation_ms: 10.0,
            speed_factor: 1.0,
            // Advanced params
            shimmer_percent: 2.0,
            breath_volume_db: -25.0,
            pitch_shift_semitones: 0.0
        };

        // Pending suggestions from AI
        this.pendingSuggestions = null;

        this.init();
    }

    init() {
        this.bindElements();
        this.bindEvents();
        this.connectWebSocket();
        this.initAudioContext();
    }

    bindElements() {
        // Header elements
        this.connectionStatus = document.getElementById('connection-status');
        this.statusText = this.connectionStatus?.querySelector('.status-text');
        this.btnUndo = document.getElementById('btn-undo');
        this.btnRedo = document.getElementById('btn-redo');
        this.btnSave = document.getElementById('btn-save');

        // Session elements
        this.presetSelect = document.getElementById('preset-select');
        this.btnStartSession = document.getElementById('btn-start-session');

        // AI elements
        this.aiDescription = document.getElementById('ai-description');
        this.btnAiSuggest = document.getElementById('btn-ai-suggest');
        this.aiSuggestions = document.getElementById('ai-suggestions');
        this.suggestionsList = document.getElementById('suggestions-list');
        this.matchedKeywords = document.getElementById('matched-keywords');
        this.btnApplySuggestions = document.getElementById('btn-apply-suggestions');
        this.btnClearSuggestions = document.getElementById('btn-clear-suggestions');

        // Preview elements
        this.previewVoice = document.getElementById('preview-voice');
        this.previewText = document.getElementById('preview-text');
        this.btnPreview = document.getElementById('btn-preview');
        this.btnPlay = document.getElementById('btn-play');
        this.btnStop = document.getElementById('btn-stop');
        this.previewStatus = document.getElementById('preview-status');
        this.waveformCanvas = document.getElementById('waveform-canvas');
        this.waveformPlaceholder = document.getElementById('waveform-placeholder');

        // Voice description
        this.voiceDescription = document.getElementById('voice-description');

        // Parameter sliders
        this.sliders = {
            formant: document.getElementById('param-formant'),
            breath: document.getElementById('param-breath'),
            jitter: document.getElementById('param-jitter'),
            drift: document.getElementById('param-drift'),
            timing: document.getElementById('param-timing'),
            speed: document.getElementById('param-speed'),
            // Advanced
            shimmer: document.getElementById('param-shimmer'),
            breathVol: document.getElementById('param-breath-vol'),
            pitchShift: document.getElementById('param-pitch-shift')
        };

        // Parameter value displays
        this.valueDisplays = {
            formant: document.getElementById('formant-value'),
            breath: document.getElementById('breath-value'),
            jitter: document.getElementById('jitter-value'),
            drift: document.getElementById('drift-value'),
            timing: document.getElementById('timing-value'),
            speed: document.getElementById('speed-value'),
            // Advanced
            shimmer: document.getElementById('shimmer-value'),
            breathVol: document.getElementById('breath-vol-value'),
            pitchShift: document.getElementById('pitch-shift-value')
        };

        this.btnResetParams = document.getElementById('btn-reset-params');

        // Save modal
        this.saveModal = document.getElementById('save-modal');
        this.saveName = document.getElementById('save-name');
        this.saveDescription = document.getElementById('save-description');
        this.saveGender = document.getElementById('save-gender');
        this.saveEnergy = document.getElementById('save-energy');
        this.saveTags = document.getElementById('save-tags');
        this.btnCloseModal = document.getElementById('btn-close-modal');
        this.btnCancelSave = document.getElementById('btn-cancel-save');
        this.btnConfirmSave = document.getElementById('btn-confirm-save');

        // Toast container
        this.toastContainer = document.getElementById('toast-container');

        // A/B Comparison elements
        this.slotA = document.getElementById('slot-a');
        this.slotB = document.getElementById('slot-b');
        this.slotAName = document.getElementById('slot-a-name');
        this.slotBName = document.getElementById('slot-b-name');
        this.btnCaptureA = document.getElementById('btn-capture-a');
        this.btnCaptureB = document.getElementById('btn-capture-b');
        this.btnPlayA = document.getElementById('btn-play-a');
        this.btnPlayB = document.getElementById('btn-play-b');
        this.btnSwapAB = document.getElementById('btn-swap-ab');
        this.btnLoadA = document.getElementById('btn-load-a');
        this.btnLoadB = document.getElementById('btn-load-b');

        // Export elements
        this.btnExport = document.getElementById('btn-export');
        this.exportDropdown = document.getElementById('export-dropdown');
        this.exportMenu = document.getElementById('export-menu');

        // Waveform seek elements
        this.waveformContainer = document.querySelector('.waveform-container');
        this.playhead = document.getElementById('playhead');

        // Tags input elements
        this.tagsContainer = document.getElementById('tags-container');
        this.tagsInput = document.getElementById('tags-input');
        this.suggestedTags = document.getElementById('suggested-tags');
        this.saveFolder = document.getElementById('save-folder');
    }

    bindEvents() {
        // Session
        this.btnStartSession?.addEventListener('click', () => this.startSession());

        // AI
        this.btnAiSuggest?.addEventListener('click', () => this.getAiSuggestions());
        this.btnApplySuggestions?.addEventListener('click', () => this.applySuggestions());
        this.btnClearSuggestions?.addEventListener('click', () => this.clearSuggestions());
        this.aiDescription?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.getAiSuggestions();
            }
        });

        // Preview
        this.btnPreview?.addEventListener('click', () => this.generatePreview());
        this.btnPlay?.addEventListener('click', () => this.playPreview());
        this.btnStop?.addEventListener('click', () => this.stopPreview());

        // Undo/Redo
        this.btnUndo?.addEventListener('click', () => this.undo());
        this.btnRedo?.addEventListener('click', () => this.redo());

        // Reset
        this.btnResetParams?.addEventListener('click', () => this.resetParams());

        // Save
        this.btnSave?.addEventListener('click', () => this.openSaveModal());
        this.btnCloseModal?.addEventListener('click', () => this.closeSaveModal());
        this.btnCancelSave?.addEventListener('click', () => this.closeSaveModal());
        this.btnConfirmSave?.addEventListener('click', () => this.savePreset());
        this.saveModal?.querySelector('.modal-backdrop')?.addEventListener('click', () => this.closeSaveModal());

        // Parameter sliders
        Object.entries(this.sliders).forEach(([key, slider]) => {
            if (slider) {
                slider.addEventListener('input', () => this.onSliderChange(key, slider.value));
                slider.addEventListener('change', () => this.commitSliderChange(key, slider.value));
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                if (e.key === 'z') {
                    e.preventDefault();
                    this.undo();
                } else if (e.key === 'y') {
                    e.preventDefault();
                    this.redo();
                } else if (e.key === 's') {
                    e.preventDefault();
                    this.openSaveModal();
                }
            }
        });

        // Keywords click to insert
        document.querySelectorAll('.keyword').forEach(kw => {
            kw.addEventListener('click', () => {
                const text = this.aiDescription.value;
                const keyword = kw.textContent;
                this.aiDescription.value = text ? `${text} ${keyword}` : keyword;
                this.aiDescription.focus();
            });
        });

        // Style chips (quick presets)
        document.querySelectorAll('.style-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const style = chip.dataset.style;
                this.aiDescription.value = style;
                this.getAiSuggestions();
                // Visual feedback
                document.querySelectorAll('.style-chip').forEach(c => c.classList.remove('active'));
                chip.classList.add('active');
            });
        });

        // Audio tags click to insert into preview text
        document.querySelectorAll('.audio-tag').forEach(tag => {
            tag.addEventListener('click', () => {
                this.insertAtCursor(this.previewText, tag.dataset.tag + ' ');
            });
        });

        // Quick insert buttons above preview text
        document.querySelectorAll('.tag-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.insertAtCursor(this.previewText, btn.dataset.insert);
            });
        });

        // Fine control mode (Shift key for sliders)
        this.fineControlActive = false;
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Shift') this.fineControlActive = true;
        });
        document.addEventListener('keyup', (e) => {
            if (e.key === 'Shift') this.fineControlActive = false;
        });

        // A/B Comparison events
        this.btnCaptureA?.addEventListener('click', () => this.captureToSlot('a'));
        this.btnCaptureB?.addEventListener('click', () => this.captureToSlot('b'));
        this.btnPlayA?.addEventListener('click', () => this.playSlot('a'));
        this.btnPlayB?.addEventListener('click', () => this.playSlot('b'));
        this.btnSwapAB?.addEventListener('click', () => this.swapABSlots());
        this.btnLoadA?.addEventListener('click', () => this.loadSlot('a'));
        this.btnLoadB?.addEventListener('click', () => this.loadSlot('b'));

        // Export dropdown events
        this.btnExport?.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleExportMenu();
        });
        document.querySelectorAll('.export-option').forEach(opt => {
            opt.addEventListener('click', () => this.exportAudio(opt.dataset.format));
        });
        document.addEventListener('click', () => this.closeExportMenu());

        // Waveform seek events
        if (this.waveformContainer) {
            this.waveformContainer.addEventListener('click', (e) => this.seekWaveform(e));
            this.waveformContainer.addEventListener('mousemove', (e) => this.updateWaveformHover(e));
            this.waveformContainer.addEventListener('mouseleave', () => this.hideWaveformHover());
            this.initWaveformHover();
        }

        // Tags input events
        this.tagsInput?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && this.tagsInput.value.trim()) {
                e.preventDefault();
                this.addTag(this.tagsInput.value.trim());
                this.tagsInput.value = '';
            } else if (e.key === 'Backspace' && !this.tagsInput.value && this.currentTags.length) {
                this.removeTag(this.currentTags.length - 1);
            }
        });
        this.tagsContainer?.addEventListener('click', () => this.tagsInput?.focus());
        document.querySelectorAll('.suggested-tag').forEach(tag => {
            tag.addEventListener('click', () => this.addTag(tag.dataset.tag));
        });
    }

    // Insert text at cursor position in textarea
    insertAtCursor(textarea, text) {
        if (!textarea) return;
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;
        const value = textarea.value;
        textarea.value = value.substring(0, start) + text + value.substring(end);
        textarea.selectionStart = textarea.selectionEnd = start + text.length;
        textarea.focus();
    }

    // ===== WebSocket =====

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.hostname}:8765`;

        this.updateStatus('Connecting...', 'connecting');

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                this.updateStatus('Connected', 'connected');
                this.showToast('Connected to Voice Studio', 'success');
                this.loadPresets();
            };

            this.ws.onclose = () => {
                this.updateStatus('Disconnected', 'disconnected');
                this.sessionId = null;
                setTimeout(() => this.connectWebSocket(), 3000);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateStatus('Error', 'error');
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (e) {
                    console.error('Failed to parse message:', e);
                }
            };
        } catch (e) {
            console.error('WebSocket connection failed:', e);
            this.updateStatus('Failed to connect', 'error');
        }
    }

    send(action, params = {}) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ action, ...params }));
        } else {
            this.showToast('Not connected to server', 'error');
        }
    }

    handleMessage(data) {
        const { action, success, error } = data;

        if (!success && error) {
            this.showToast(error, 'error');
            return;
        }

        switch (action) {
            case 'connected':
                console.log('Connected:', data.data);
                break;

            case 'studio_start':
                this.onSessionStarted(data.data);
                break;

            case 'studio_adjust':
                this.onParamsAdjusted(data.data);
                break;

            case 'studio_ai_describe':
                this.onAiSuggestions(data.data);
                break;

            case 'studio_preview':
                this.onPreviewGenerated(data.data);
                break;

            case 'studio_undo':
            case 'studio_redo':
                this.onStateRestored(data.data);
                break;

            case 'studio_status':
                this.onStatusReceived(data.data);
                break;

            case 'studio_save':
                this.onPresetSaved(data.data);
                break;

            case 'studio_list_presets':
                this.onPresetsLoaded(data.data);
                break;
        }
    }

    updateStatus(text, state) {
        if (this.statusText) {
            this.statusText.textContent = text;
        }
        if (this.connectionStatus) {
            this.connectionStatus.className = `connection-badge ${state}`;
        }
    }

    // ===== Session Management =====

    loadPresets() {
        this.send('studio_list_presets');
    }

    onPresetsLoaded(data) {
        if (!this.presetSelect) return;

        // Clear existing options except first
        while (this.presetSelect.options.length > 1) {
            this.presetSelect.remove(1);
        }

        // Group presets by source
        const grouped = {};
        data.presets.forEach(preset => {
            const source = preset.id.split(':')[0];
            if (!grouped[source]) grouped[source] = [];
            grouped[source].push(preset);
        });

        // Add grouped options
        Object.entries(grouped).forEach(([source, presets]) => {
            const group = document.createElement('optgroup');
            group.label = source.charAt(0).toUpperCase() + source.slice(1);

            presets.forEach(preset => {
                const option = document.createElement('option');
                option.value = preset.id;
                option.textContent = preset.name;
                group.appendChild(option);
            });

            this.presetSelect.appendChild(group);
        });
    }

    startSession() {
        const basePresetId = this.presetSelect?.value || null;

        this.send('studio_start', {
            base_preset_id: basePresetId || undefined,
            preview_voice: this.previewVoice?.value
        });

        this.setPreviewStatus('Starting session...', 'loading');
    }

    onSessionStarted(data) {
        this.sessionId = data.session_id;
        this.updateParams(data.current_params);
        this.updateUndoRedoButtons(data.status.can_undo, data.status.can_redo);

        // Enable controls
        if (this.btnSave) this.btnSave.disabled = false;
        if (this.btnPreview) this.btnPreview.disabled = false;

        this.showToast('Session started!', 'success');
        this.setPreviewStatus('Ready to generate preview', 'success');
        this.updateVoiceDescription();
    }

    // ===== Parameter Management =====

    updateParams(params) {
        this.params = { ...this.params, ...params };

        // Update sliders
        if (this.sliders.formant) {
            this.sliders.formant.value = params.formant_ratio ?? 1.0;
            this.updateValueDisplay('formant', params.formant_ratio ?? 1.0);
        }
        if (this.sliders.breath) {
            this.sliders.breath.value = params.breath_intensity ?? 0.15;
            this.updateValueDisplay('breath', params.breath_intensity ?? 0.15);
        }
        if (this.sliders.jitter) {
            this.sliders.jitter.value = params.jitter_percent ?? 0.5;
            this.updateValueDisplay('jitter', params.jitter_percent ?? 0.5);
        }
        if (this.sliders.drift) {
            this.sliders.drift.value = params.pitch_drift_cents ?? 8.0;
            this.updateValueDisplay('drift', params.pitch_drift_cents ?? 8.0);
        }
        if (this.sliders.timing) {
            this.sliders.timing.value = params.timing_variation_ms ?? 10.0;
            this.updateValueDisplay('timing', params.timing_variation_ms ?? 10.0);
        }
        if (this.sliders.speed) {
            this.sliders.speed.value = params.speed_factor ?? 1.0;
            this.updateValueDisplay('speed', params.speed_factor ?? 1.0);
        }

        // Advanced params
        if (this.sliders.shimmer) {
            this.sliders.shimmer.value = params.shimmer_percent ?? 2.0;
            this.updateValueDisplay('shimmer', params.shimmer_percent ?? 2.0);
        }
        if (this.sliders.breathVol) {
            this.sliders.breathVol.value = params.breath_volume_db ?? -25.0;
            this.updateValueDisplay('breathVol', params.breath_volume_db ?? -25.0);
        }
        if (this.sliders.pitchShift) {
            this.sliders.pitchShift.value = params.pitch_shift_semitones ?? 0.0;
            this.updateValueDisplay('pitchShift', params.pitch_shift_semitones ?? 0.0);
        }

        this.updateVoiceDescription();
    }

    updateValueDisplay(key, value) {
        const display = this.valueDisplays[key];
        if (!display) return;

        switch (key) {
            case 'formant':
                display.textContent = value.toFixed(2);
                break;
            case 'breath':
                display.textContent = value.toFixed(2);
                break;
            case 'jitter':
                display.textContent = `${value.toFixed(2)}%`;
                break;
            case 'drift':
                display.textContent = `${value.toFixed(1)} ct`;
                break;
            case 'timing':
                display.textContent = `${Math.round(value)} ms`;
                break;
            case 'speed':
                display.textContent = `${value.toFixed(2)}x`;
                break;
            // Advanced params
            case 'shimmer':
                display.textContent = `${value.toFixed(1)}%`;
                break;
            case 'breathVol':
                display.textContent = `${Math.round(value)} dB`;
                break;
            case 'pitchShift':
                display.textContent = `${value > 0 ? '+' : ''}${Math.round(value)} st`;
                break;
        }
    }

    onSliderChange(key, value) {
        value = parseFloat(value);
        this.updateValueDisplay(key, value);
    }

    commitSliderChange(key, value) {
        if (!this.sessionId) {
            this.showToast('Start a session first', 'warning');
            return;
        }

        value = parseFloat(value);
        const paramMap = {
            formant: 'formant_ratio',
            breath: 'breath_intensity',
            jitter: 'jitter_percent',
            drift: 'pitch_drift_cents',
            timing: 'timing_variation_ms',
            speed: 'speed_factor',
            // Advanced
            shimmer: 'shimmer_percent',
            breathVol: 'breath_volume_db',
            pitchShift: 'pitch_shift_semitones'
        };

        const paramName = paramMap[key];
        if (paramName) {
            this.send('studio_adjust', { [paramName]: value });
        }
    }

    onParamsAdjusted(data) {
        this.updateParams(data.current_params);
        this.updateUndoRedoButtons(data.can_undo, data.can_redo);
    }

    resetParams() {
        if (!this.sessionId) return;

        this.send('studio_adjust', {
            formant_ratio: 1.0,
            breath_intensity: 0.15,
            jitter_percent: 0.5,
            pitch_drift_cents: 8.0,
            timing_variation_ms: 10.0,
            speed_factor: 1.0
        });
    }

    updateVoiceDescription() {
        if (!this.voiceDescription) return;

        const parts = [];

        // Formant
        if (this.params.formant_ratio < 0.92) {
            parts.push('Very deep');
        } else if (this.params.formant_ratio < 0.97) {
            parts.push('Deep');
        } else if (this.params.formant_ratio > 1.08) {
            parts.push('Very bright');
        } else if (this.params.formant_ratio > 1.03) {
            parts.push('Bright');
        }

        // Breath
        if (this.params.breath_intensity > 0.3) {
            parts.push('very breathy');
        } else if (this.params.breath_intensity > 0.2) {
            parts.push('breathy');
        }

        // Jitter
        if (this.params.jitter_percent > 1.5) {
            parts.push('gravelly');
        } else if (this.params.jitter_percent > 0.8) {
            parts.push('textured');
        } else if (this.params.jitter_percent < 0.3) {
            parts.push('smooth');
        }

        // Speed
        if (this.params.speed_factor < 0.85) {
            parts.push('slow and deliberate');
        } else if (this.params.speed_factor > 1.2) {
            parts.push('quick and energetic');
        }

        // Pitch drift
        if (this.params.pitch_drift_cents > 15) {
            parts.push('very expressive');
        } else if (this.params.pitch_drift_cents < 4) {
            parts.push('steady pitch');
        }

        if (parts.length === 0) {
            this.voiceDescription.innerHTML = '<span class="placeholder-text">Neutral, balanced voice</span>';
        } else {
            this.voiceDescription.textContent = parts.join(', ').replace(/^./, c => c.toUpperCase());
        }
    }

    // ===== AI Suggestions =====

    getAiSuggestions() {
        const description = this.aiDescription?.value?.trim();
        if (!description) {
            this.showToast('Enter a voice description', 'warning');
            return;
        }

        if (!this.sessionId) {
            this.showToast('Start a session first', 'warning');
            return;
        }

        this.send('studio_ai_describe', { description, apply: false });
        this.btnAiSuggest.disabled = true;
        this.btnAiSuggest.innerHTML = `
            <svg class="spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10" stroke-dasharray="60" stroke-dashoffset="20"/>
            </svg>
            Analyzing...
        `;
    }

    onAiSuggestions(data) {
        this.btnAiSuggest.disabled = false;
        this.btnAiSuggest.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
            </svg>
            Get AI Suggestions
        `;

        const { result } = data;

        if (!result.params || Object.keys(result.params).length === 0) {
            this.showToast('No matching voice qualities found. Try keywords like: deep, warm, breathy', 'warning');
            return;
        }

        this.pendingSuggestions = result.params;

        // Show matched keywords
        if (this.matchedKeywords) {
            this.matchedKeywords.textContent = result.matched_keywords.join(', ');
        }

        // Build suggestions list
        if (this.suggestionsList) {
            this.suggestionsList.innerHTML = '';

            const paramNames = {
                formant_ratio: 'Voice Depth',
                breath_intensity: 'Breathiness',
                jitter_percent: 'Texture',
                pitch_drift_cents: 'Pitch Variation',
                timing_variation_ms: 'Timing',
                speed_factor: 'Speed'
            };

            Object.entries(result.params).forEach(([param, value]) => {
                const item = document.createElement('div');
                item.className = 'suggestion-item';
                item.innerHTML = `
                    <span class="param-name">${paramNames[param] || param}</span>
                    <span class="param-change">${typeof value === 'number' ? value.toFixed(2) : value}</span>
                `;
                this.suggestionsList.appendChild(item);
            });
        }

        // Show panel
        if (this.aiSuggestions) {
            this.aiSuggestions.classList.remove('hidden');
        }
    }

    applySuggestions() {
        if (!this.pendingSuggestions || !this.sessionId) return;

        this.send('studio_adjust', this.pendingSuggestions);
        this.clearSuggestions();
        this.showToast('Applied AI suggestions', 'success');
    }

    clearSuggestions() {
        this.pendingSuggestions = null;
        if (this.aiSuggestions) {
            this.aiSuggestions.classList.add('hidden');
        }
    }

    // ===== Undo/Redo =====

    undo() {
        if (!this.sessionId) return;
        this.send('studio_undo');
    }

    redo() {
        if (!this.sessionId) return;
        this.send('studio_redo');
    }

    onStateRestored(data) {
        this.updateParams(data.current_params);
        this.updateUndoRedoButtons(data.can_undo, data.can_redo);
    }

    updateUndoRedoButtons(canUndo, canRedo) {
        if (this.btnUndo) this.btnUndo.disabled = !canUndo;
        if (this.btnRedo) this.btnRedo.disabled = !canRedo;
    }

    onStatusReceived(data) {
        if (data.active && data.current_params) {
            this.sessionId = data.session.session_id;
            this.updateParams(data.current_params);
            this.updateUndoRedoButtons(data.session.can_undo, data.session.can_redo);
        }
    }

    // ===== Preview =====

    initAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) {
            console.warn('AudioContext not available:', e);
        }
    }

    generatePreview() {
        if (!this.sessionId) {
            this.showToast('Start a session first', 'warning');
            return;
        }

        const text = this.previewText?.value?.trim();
        if (!text) {
            this.showToast('Enter preview text', 'warning');
            return;
        }

        this.setPreviewStatus('Generating preview...', 'loading');
        this.btnPreview.disabled = true;

        this.send('studio_preview', {
            text,
            voice: this.previewVoice?.value,
            return_audio: true
        });
    }

    onPreviewGenerated(data) {
        this.btnPreview.disabled = false;

        if (data.audio_base64) {
            // Convert base64 to audio buffer
            const binaryString = atob(data.audio_base64);
            const bytes = new Float32Array(binaryString.length / 4);
            const dataView = new DataView(new ArrayBuffer(binaryString.length));

            for (let i = 0; i < binaryString.length; i++) {
                dataView.setUint8(i, binaryString.charCodeAt(i));
            }

            for (let i = 0; i < bytes.length; i++) {
                bytes[i] = dataView.getFloat32(i * 4, true);
            }

            this.previewAudioData = {
                samples: bytes,
                sampleRate: data.sample_rate
            };

            // Draw waveform
            this.drawWaveform(bytes);

            // Enable play and export buttons
            if (this.btnPlay) this.btnPlay.disabled = false;
            if (this.btnExport) this.btnExport.disabled = false;

            this.setPreviewStatus(`Preview ready (${data.duration.toFixed(1)}s)`, 'success');
        } else {
            this.setPreviewStatus('Preview generated (file saved)', 'success');
        }
    }

    drawWaveform(samples) {
        if (!this.waveformCanvas) return;

        // Hide placeholder
        if (this.waveformPlaceholder) {
            this.waveformPlaceholder.classList.add('hidden');
        }

        const canvas = this.waveformCanvas;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;

        // Set canvas size
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const width = rect.width;
        const height = rect.height;
        const centerY = height / 2;

        // Clear
        ctx.fillStyle = '#0f0f14';
        ctx.fillRect(0, 0, width, height);

        // Draw center line
        ctx.strokeStyle = '#2d2d3a';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        ctx.lineTo(width, centerY);
        ctx.stroke();

        // Draw waveform
        const step = Math.ceil(samples.length / width);
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        for (let i = 0; i < width; i++) {
            const start = i * step;
            const end = Math.min(start + step, samples.length);

            let min = 1;
            let max = -1;
            for (let j = start; j < end; j++) {
                if (samples[j] < min) min = samples[j];
                if (samples[j] > max) max = samples[j];
            }

            const y1 = centerY + min * (height * 0.4);
            const y2 = centerY + max * (height * 0.4);

            if (i === 0) {
                ctx.moveTo(i, y1);
            }
            ctx.lineTo(i, y1);
            ctx.lineTo(i, y2);
        }

        ctx.stroke();

        // Add gradient overlay
        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, 'rgba(99, 102, 241, 0.1)');
        gradient.addColorStop(0.5, 'rgba(99, 102, 241, 0)');
        gradient.addColorStop(1, 'rgba(99, 102, 241, 0.1)');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
    }

    playPreview() {
        if (!this.previewAudioData || !this.audioContext) return;

        this.stopPreview();

        const { samples, sampleRate } = this.previewAudioData;
        const buffer = this.audioContext.createBuffer(1, samples.length, sampleRate);
        buffer.copyToChannel(samples, 0);

        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioContext.destination);

        // Track playback for playhead animation
        this.playbackOffset = 0;
        this.playbackStartTime = this.audioContext.currentTime;
        const duration = samples.length / sampleRate;

        source.onended = () => {
            this.isPlaying = false;
            this.hidePlayhead();
            if (this.btnStop) this.btnStop.disabled = true;
        };

        source.start(0);
        this.currentAudio = source;
        this.isPlaying = true;

        if (this.btnStop) this.btnStop.disabled = false;
        this.startPlayheadAnimation(duration);
    }

    stopPreview() {
        if (this.currentAudio) {
            try {
                this.currentAudio.stop();
            } catch (e) {}
            this.currentAudio = null;
        }
        this.isPlaying = false;
        this.hidePlayhead();
        if (this.btnStop) this.btnStop.disabled = true;
    }

    setPreviewStatus(text, type = '') {
        if (this.previewStatus) {
            this.previewStatus.textContent = text;
            this.previewStatus.className = `preview-status ${type}`;
        }
    }

    // ===== Save =====

    openSaveModal() {
        if (!this.sessionId) {
            this.showToast('Start a session first', 'warning');
            return;
        }
        if (this.saveModal) {
            this.saveModal.classList.remove('hidden');
        }
    }

    closeSaveModal() {
        if (this.saveModal) {
            this.saveModal.classList.add('hidden');
        }
    }

    savePreset() {
        const name = this.saveName?.value?.trim();
        if (!name) {
            this.showToast('Enter a preset name', 'warning');
            return;
        }

        const folder = this.saveFolder?.value || '';
        const tags = [...this.currentTags]; // Use the enhanced tags array

        this.send('studio_save', {
            name,
            folder,
            description: this.saveDescription?.value?.trim() || '',
            gender: this.saveGender?.value || null,
            energy: this.saveEnergy?.value || 'neutral',
            tags
        });

        this.closeSaveModal();
        this.clearTags();
    }

    onPresetSaved(data) {
        this.showToast(`Preset "${data.name}" saved!`, 'success');
        this.loadPresets(); // Refresh list
    }

    // ===== Toast Notifications =====

    showToast(message, type = 'info') {
        if (!this.toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;

        this.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // ===== A/B Comparison =====

    captureToSlot(slot) {
        if (!this.sessionId) {
            this.showToast('Start a session first', 'warning');
            return;
        }

        if (!this.previewAudioData) {
            this.showToast('Generate a preview first', 'warning');
            return;
        }

        // Capture current state
        this.abSlots[slot] = {
            params: { ...this.params },
            audio: { ...this.previewAudioData },
            name: this.generateSlotName(),
            voice: this.previewVoice?.value || 'af_bella'
        };

        // Update UI
        this.updateSlotUI(slot);
        this.showToast(`Captured to slot ${slot.toUpperCase()}`, 'success');
    }

    generateSlotName() {
        const parts = [];
        if (this.params.formant_ratio < 0.95) parts.push('Deep');
        else if (this.params.formant_ratio > 1.05) parts.push('Bright');
        if (this.params.breath_intensity > 0.25) parts.push('Breathy');
        if (this.params.speed_factor < 0.9) parts.push('Slow');
        else if (this.params.speed_factor > 1.1) parts.push('Fast');
        return parts.length > 0 ? parts.join(', ') : 'Default';
    }

    updateSlotUI(slot) {
        const slotEl = slot === 'a' ? this.slotA : this.slotB;
        const nameEl = slot === 'a' ? this.slotAName : this.slotBName;
        const playBtn = slot === 'a' ? this.btnPlayA : this.btnPlayB;
        const data = this.abSlots[slot];

        if (data.audio) {
            slotEl?.classList.add('has-audio');
            if (nameEl) nameEl.textContent = data.name;
            if (playBtn) playBtn.disabled = false;
        } else {
            slotEl?.classList.remove('has-audio');
            if (nameEl) nameEl.textContent = 'Empty';
            if (playBtn) playBtn.disabled = true;
        }
    }

    playSlot(slot) {
        const data = this.abSlots[slot];
        if (!data.audio || !this.audioContext) return;

        this.stopPreview();

        const { samples, sampleRate } = data.audio;
        const buffer = this.audioContext.createBuffer(1, samples.length, sampleRate);
        buffer.copyToChannel(samples, 0);

        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioContext.destination);

        source.onended = () => {
            this.isPlaying = false;
        };

        source.start(0);
        this.currentAudio = source;
        this.isPlaying = true;

        this.showToast(`Playing ${slot.toUpperCase()}`, 'info');
    }

    swapABSlots() {
        const temp = this.abSlots.a;
        this.abSlots.a = this.abSlots.b;
        this.abSlots.b = temp;
        this.updateSlotUI('a');
        this.updateSlotUI('b');
        this.showToast('Swapped A and B', 'info');
    }

    loadSlot(slot) {
        const data = this.abSlots[slot];
        if (!data.params) {
            this.showToast(`Slot ${slot.toUpperCase()} is empty`, 'warning');
            return;
        }

        // Apply the captured parameters
        this.send('studio_adjust', data.params);
        this.showToast(`Loaded ${slot.toUpperCase()} settings`, 'success');
    }

    // ===== Export =====

    toggleExportMenu() {
        this.exportDropdown?.classList.toggle('open');
        this.exportMenu?.classList.toggle('hidden');
    }

    closeExportMenu() {
        this.exportDropdown?.classList.remove('open');
        this.exportMenu?.classList.add('hidden');
    }

    async exportAudio(format) {
        this.closeExportMenu();

        if (!this.previewAudioData) {
            this.showToast('Generate a preview first', 'warning');
            return;
        }

        this.showToast(`Exporting as ${format.toUpperCase()}...`, 'info');

        try {
            const { samples, sampleRate } = this.previewAudioData;
            let blob;

            if (format === 'wav') {
                blob = this.encodeWav(samples, sampleRate);
            } else {
                // For other formats, send to server for encoding
                this.send('studio_export', {
                    format,
                    audio_base64: this.floatArrayToBase64(samples),
                    sample_rate: sampleRate
                });
                return;
            }

            // Download the blob
            this.downloadBlob(blob, `voice_preview.${format}`);
            this.showToast(`Exported as ${format.toUpperCase()}`, 'success');
        } catch (e) {
            this.showToast('Export failed: ' + e.message, 'error');
        }
    }

    encodeWav(samples, sampleRate) {
        const numChannels = 1;
        const bitsPerSample = 16;
        const bytesPerSample = bitsPerSample / 8;
        const blockAlign = numChannels * bytesPerSample;

        const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
        const view = new DataView(buffer);

        const writeString = (offset, str) => {
            for (let i = 0; i < str.length; i++) {
                view.setUint8(offset + i, str.charCodeAt(i));
            }
        };

        // WAV header
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + samples.length * bytesPerSample, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);
        writeString(36, 'data');
        view.setUint32(40, samples.length * bytesPerSample, true);

        // Convert float samples to 16-bit PCM
        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            offset += 2;
        }

        return new Blob([buffer], { type: 'audio/wav' });
    }

    floatArrayToBase64(float32Array) {
        const uint8Array = new Uint8Array(float32Array.buffer);
        let binary = '';
        for (let i = 0; i < uint8Array.length; i++) {
            binary += String.fromCharCode(uint8Array[i]);
        }
        return btoa(binary);
    }

    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // ===== Waveform Seek =====

    initWaveformHover() {
        // Create hover line and tooltip
        this.hoverLine = document.createElement('div');
        this.hoverLine.className = 'waveform-hover-line';
        this.waveformContainer?.appendChild(this.hoverLine);

        this.timeTooltip = document.createElement('div');
        this.timeTooltip.className = 'waveform-time-tooltip';
        this.waveformContainer?.appendChild(this.timeTooltip);
    }

    updateWaveformHover(e) {
        if (!this.previewAudioData || !this.hoverLine || !this.timeTooltip) return;

        const rect = this.waveformContainer.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percent = x / rect.width;
        const duration = this.previewAudioData.samples.length / this.previewAudioData.sampleRate;
        const time = percent * duration;

        this.hoverLine.style.left = `${x}px`;
        this.timeTooltip.style.left = `${x}px`;
        this.timeTooltip.textContent = this.formatTime(time);
    }

    hideWaveformHover() {
        // Opacity is handled by CSS :hover
    }

    seekWaveform(e) {
        if (!this.previewAudioData || !this.audioContext) return;

        const rect = this.waveformContainer.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percent = x / rect.width;
        const duration = this.previewAudioData.samples.length / this.previewAudioData.sampleRate;
        const seekTime = percent * duration;

        // Stop current playback
        this.stopPreview();

        // Start playback from seek position
        const { samples, sampleRate } = this.previewAudioData;
        const startSample = Math.floor(seekTime * sampleRate);
        const remainingSamples = samples.slice(startSample);

        if (remainingSamples.length === 0) return;

        const buffer = this.audioContext.createBuffer(1, remainingSamples.length, sampleRate);
        buffer.copyToChannel(remainingSamples, 0);

        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioContext.destination);

        // Track playback for playhead animation
        this.playbackOffset = seekTime;
        this.playbackStartTime = this.audioContext.currentTime;

        source.onended = () => {
            this.isPlaying = false;
            this.hidePlayhead();
            if (this.btnStop) this.btnStop.disabled = true;
        };

        source.start(0);
        this.currentAudio = source;
        this.isPlaying = true;

        if (this.btnStop) this.btnStop.disabled = false;
        this.startPlayheadAnimation(duration);
    }

    startPlayheadAnimation(duration) {
        if (!this.playhead || !this.waveformContainer) return;

        this.playhead.classList.remove('hidden');

        const animate = () => {
            if (!this.isPlaying) {
                this.hidePlayhead();
                return;
            }

            const elapsed = this.audioContext.currentTime - this.playbackStartTime;
            const currentTime = this.playbackOffset + elapsed;
            const percent = currentTime / duration;

            if (percent >= 1) {
                this.hidePlayhead();
                return;
            }

            const rect = this.waveformContainer.getBoundingClientRect();
            this.playhead.style.left = `${percent * rect.width}px`;

            requestAnimationFrame(animate);
        };

        requestAnimationFrame(animate);
    }

    hidePlayhead() {
        this.playhead?.classList.add('hidden');
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 10);
        return `${mins}:${secs.toString().padStart(2, '0')}.${ms}`;
    }

    // ===== Tags Input =====

    addTag(tagText) {
        const tag = tagText.toLowerCase().trim();
        if (!tag || this.currentTags.includes(tag)) return;

        this.currentTags.push(tag);
        this.renderTags();
    }

    removeTag(index) {
        this.currentTags.splice(index, 1);
        this.renderTags();
    }

    clearTags() {
        this.currentTags = [];
        this.renderTags();
    }

    renderTags() {
        if (!this.tagsContainer) return;

        // Remove existing tag elements (keep the input)
        this.tagsContainer.querySelectorAll('.preset-tag').forEach(el => el.remove());

        // Add tag elements before the input
        this.currentTags.forEach((tag, index) => {
            const tagEl = document.createElement('span');
            tagEl.className = 'preset-tag';
            tagEl.innerHTML = `
                ${tag}
                <button class="remove-tag" data-index="${index}">&times;</button>
            `;
            tagEl.querySelector('.remove-tag').addEventListener('click', (e) => {
                e.stopPropagation();
                this.removeTag(index);
            });
            this.tagsContainer.insertBefore(tagEl, this.tagsInput);
        });
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.voiceStudio = new VoiceStudio();
});

// Add spin animation
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .spin {
        animation: spin 1s linear infinite;
    }
`;
document.head.appendChild(style);
