# Voice Studio Roadmap

## Current State (v1.0) ✅

### Features Delivered
- **Dark theme UI** with 3-column layout
- **Real-time WebSocket** connection with status indicator
- **6 base voices** (Bella, Sarah, Michael, Adam, Emma, George)
- **6 acoustic parameters** with intuitive sliders:
  - Voice Depth (formant ratio)
  - Breathiness
  - Voice Texture (jitter)
  - Pitch Variation (drift)
  - Timing Feel
  - Speaking Speed
- **AI Description** - natural language to parameters
- **Waveform visualization** with canvas
- **Audio preview** with play/stop controls
- **Undo/Redo** with keyboard shortcuts (Ctrl+Z/Y)
- **Save presets** with metadata
- **71 base presets** from catalog

---

## Phase 2: Enhanced Voice Selection

### 2.1 Expanded Voice Library
- [ ] Add all Kokoro voices (currently 6 of ~20+)
- [ ] Voice preview thumbnails (short audio samples)
- [ ] Voice categorization: Gender, Accent, Style
- [ ] Favorite voices with localStorage persistence
- [ ] Recently used voices list

### 2.2 Voice Comparison
- [ ] A/B comparison mode (listen to two voices side-by-side)
- [ ] "Find similar voices" based on acoustic properties
- [ ] Voice compatibility hints (which voices blend well)

### 2.3 Voice Cloning Integration
- [ ] Upload reference audio for voice cloning
- [ ] Clone voice quality indicator
- [ ] Cloned voice management (rename, delete, export)

---

## Phase 3: Advanced Parameters Section

### 3.1 Collapsible Advanced Panel
- [ ] Toggle between Basic/Advanced modes
- [ ] Preserve user preference in localStorage

### 3.2 Additional Acoustic Parameters
- [ ] **Shimmer** (amplitude variation) - adds roughness
- [ ] **Breathiness location** - where breaths appear
- [ ] **Pitch contour** - rising/falling intonation patterns
- [ ] **Emphasis patterns** - stress on certain syllables
- [ ] **Pause insertion** - natural pauses between phrases

### 3.3 Prosody Controls
- [ ] **Emotion intensity** slider (0-100%)
- [ ] **Speaking style** presets (conversational, formal, excited)
- [ ] **Sentence-level pacing** - vary speed within phrases

### 3.4 Post-Processing Effects
- [ ] **Reverb** - room size, wet/dry mix
- [ ] **EQ** - bass/mid/treble adjustment
- [ ] **Compression** - dynamic range control
- [ ] **De-esser** - reduce sibilance

---

## Phase 4: Smarter AI Assistant

### 4.1 Enhanced Natural Language Understanding
- [ ] Multi-word phrase recognition ("warm and friendly narrator")
- [ ] Negation support ("not too breathy", "less robotic")
- [ ] Comparative adjustments ("a bit deeper", "much faster")
- [ ] Context-aware suggestions based on use case

### 4.2 AI-Powered Features
- [ ] **Auto-suggest** parameters based on text content
- [ ] **Style transfer** - "make it sound like a podcast host"
- [ ] **Emotion detection** from preview text
- [ ] **Quality scoring** - rate naturalness of current settings

### 4.3 Prompt Templates
- [ ] Pre-built prompts: "audiobook narrator", "news anchor", etc.
- [ ] User-saved prompt favorites
- [ ] Prompt history with quick recall

### 4.4 Interactive Tuning
- [ ] "More like this" / "Less like this" feedback buttons
- [ ] AI learns from user adjustments
- [ ] Suggest next parameter to tweak

---

## Phase 5: Workflow & Productivity

### 5.1 Preset Management
- [ ] Folder organization for presets
- [ ] Tags and search/filter
- [ ] Import/export presets (JSON)
- [ ] Preset sharing (generate shareable link)

### 5.2 Batch Processing
- [ ] Queue multiple text samples
- [ ] Apply same voice to multiple texts
- [ ] Bulk export with naming patterns

### 5.3 Project Mode
- [ ] Save entire sessions (text + voice + settings)
- [ ] Version history for presets
- [ ] Collaboration features (share project)

### 5.4 Keyboard Power User Mode
- [ ] Full keyboard navigation
- [ ] Customizable hotkeys
- [ ] Command palette (Ctrl+K)

---

## Phase 6: Visual & Audio Enhancements

### 6.1 Waveform Improvements
- [ ] Zoomable waveform
- [ ] Playhead position indicator
- [ ] Click-to-seek functionality
- [ ] Spectrogram view toggle

### 6.2 Real-time Visualization
- [ ] Live audio levels during playback
- [ ] Frequency spectrum analyzer
- [ ] Pitch contour overlay

### 6.3 UI Polish
- [ ] Light theme option
- [ ] Customizable accent colors
- [ ] Compact/expanded layout modes
- [ ] Mobile-responsive design improvements

---

## Phase 7: Integration & Export

### 7.1 Export Options
- [ ] Multiple audio formats (WAV, MP3, OGG, FLAC)
- [ ] Quality/bitrate selection
- [ ] Metadata embedding (title, artist, etc.)

### 7.2 External Integrations
- [ ] Copy audio to clipboard
- [ ] Direct upload to cloud storage
- [ ] API endpoint for programmatic access
- [ ] Webhook notifications on generation complete

### 7.3 Accessibility
- [ ] Screen reader support
- [ ] High contrast mode
- [ ] Reduced motion option

---

## Technical Debt & Performance

### Optimization
- [ ] Audio caching (don't regenerate unchanged previews)
- [ ] Lazy-load voice models on demand
- [ ] WebSocket reconnection improvements
- [ ] Service worker for offline parameter editing

### Code Quality
- [ ] TypeScript migration for studio.js
- [ ] Component extraction (React/Vue consideration)
- [ ] E2E tests with Playwright
- [ ] Performance benchmarks

---

## Priority Matrix

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| More Kokoro voices | High | Low | 🔥 P1 |
| Advanced params section | High | Medium | 🔥 P1 |
| AI prompt templates | High | Low | 🔥 P1 |
| Preset folders/tags | Medium | Medium | P2 |
| Voice comparison A/B | High | Medium | P2 |
| Waveform zoom/seek | Medium | Medium | P2 |
| Export format options | Medium | Low | P2 |
| Voice cloning UI | High | High | P3 |
| Batch processing | Medium | High | P3 |
| Real-time spectrum | Low | Medium | P3 |

---

## Next Steps (Recommended)

1. **Immediate wins** (can ship in 1-2 sessions):
   - Add remaining Kokoro voices to dropdown
   - Create collapsible "Advanced" section with shimmer + more params
   - Add 5-10 AI prompt templates

2. **Short-term** (next few sessions):
   - Implement preset folders and search
   - Add A/B voice comparison
   - Waveform interactivity (seek, zoom)

3. **Medium-term**:
   - Voice cloning integration
   - Batch processing workflow
   - Mobile layout polish
