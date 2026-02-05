# Vocal Biomarkers

Using voice analysis for health assessment and monitoring.

## Overview

**Vocal biomarkers** are measurable voice characteristics that indicate health conditions. The voice reflects the state of multiple body systems:

- **Neurological:** Motor control, cognitive function
- **Respiratory:** Lung capacity, airway health
- **Cardiovascular:** Blood oxygenation, heart function
- **Psychological:** Mood, stress, mental health
- **Endocrine:** Hormonal balance

> "Voice is the new blood" - describing non-invasive health monitoring potential

## Market & Trends (2026)

- **Market size:** $1.9B (2021) → projected $5B+ (2028)
- **Key players:** Canary Speech, Sonde Health, Kintsugi, Winterlight Labs
- **FDA status:** No approved diagnostics yet, but clinical trials advancing

### 2026 Predictions (Canary Speech)
1. **Ambient listening** for ADHD/autism screening in pediatric visits
2. **Virtual care integration** with HIPAA-compliant platforms
3. **Continuous monitoring** via smart devices
4. **Mental health screening** in routine telehealth

## Detectable Conditions

### Neurological Disorders

| Condition | Voice Markers | Detection Accuracy |
|-----------|---------------|-------------------|
| **Parkinson's** | Reduced loudness, monotone, tremor, imprecise consonants | 85-95% |
| **Alzheimer's** | Pauses, word-finding difficulty, reduced complexity | 80-90% |
| **ALS** | Slurred speech, hypernasality, reduced rate | High correlation |
| **Multiple Sclerosis** | Scanning speech, tremor, fatigue patterns | Emerging |
| **Stroke** | Asymmetric articulation, prosody changes | Post-event detection |

**Key Features:**
- Jitter/shimmer elevation
- F0 variability reduction
- Speaking rate changes
- Pause patterns

### Mental Health

| Condition | Voice Markers | Notes |
|-----------|---------------|-------|
| **Depression** | Lower pitch, reduced range, slower tempo, breathy | 71% sensitivity, 73% specificity |
| **Anxiety** | Higher pitch, faster rate, more pauses | Correlates with arousal |
| **PTSD** | Flat affect, hesitations, specific prosody patterns | Emerging research |
| **Bipolar (manic)** | Rapid speech, higher pitch, increased volume | State-dependent |
| **Schizophrenia** | Flat prosody, unusual pauses, reduced coherence | Negative symptoms |

### Respiratory Conditions

| Condition | Voice Markers |
|-----------|---------------|
| **COVID-19** | Changed resonance, breathiness, fatigue patterns |
| **Asthma** | Breathiness, reduced phrase length |
| **COPD** | Strained quality, breath patterns |
| **Sleep Apnea** | Morning hoarseness, fatigue markers |

### Cardiovascular

- **Heart Failure:** Vocal fatigue, breathiness
- **Coronary Artery Disease:** Subtle acoustic changes (research phase)

### Other Applications

- **Pain Assessment:** Vocal strain, pitch changes
- **Diabetes:** Subtle acoustic markers (research)
- **Thyroid Disorders:** Pitch changes, hoarseness
- **Hydration Status:** Voice quality changes

## Acoustic Features for Biomarkers

### Standard Voice Quality Metrics

| Feature | Extraction | Clinical Relevance |
|---------|------------|-------------------|
| **F0 (Pitch)** | Autocorrelation, RAPT | Depression, Parkinson's |
| **F0 Variability** | Std dev of F0 | Emotional state, motor control |
| **Jitter** | Cycle-to-cycle F0 variation | Neurological health |
| **Shimmer** | Cycle-to-cycle amplitude variation | Vocal fold function |
| **HNR** | Harmonic vs noise energy | Overall voice quality |
| **Speaking Rate** | Words/syllables per second | Cognitive load, mood |

### Advanced Features

| Feature | Description | Use |
|---------|-------------|-----|
| **MFCC** | Mel-frequency cepstral coefficients | General voice fingerprint |
| **Formants** | Vocal tract resonances | Articulation health |
| **Voice Onset Time** | Consonant timing | Motor control |
| **Pause Patterns** | Duration, frequency of silence | Cognitive function |
| **Spectral Centroid** | Brightness measure | Emotional state |
| **CPP** | Cepstral peak prominence | Dysphonia severity |

### Machine Learning Features

```python
# Example feature extraction for biomarker analysis
import librosa
import numpy as np

def extract_biomarker_features(audio_path):
    y, sr = librosa.load(audio_path)

    features = {}

    # Pitch features
    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500)
    f0_clean = f0[~np.isnan(f0)]
    features['f0_mean'] = np.mean(f0_clean)
    features['f0_std'] = np.std(f0_clean)
    features['f0_range'] = np.ptp(f0_clean)

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i])

    # Energy features
    rms = librosa.feature.rms(y=y)
    features['energy_mean'] = np.mean(rms)
    features['energy_std'] = np.std(rms)

    # Tempo/rhythm
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo

    return features
```

## Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                  VOCAL BIOMARKER PIPELINE                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Audio Input                                                 │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────────┐                                         │
│  │ Preprocessing   │  Noise reduction, VAD, normalization    │
│  └────────┬────────┘                                         │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │ Feature         │  F0, jitter, shimmer, MFCCs, etc.       │
│  │ Extraction      │                                         │
│  └────────┬────────┘                                         │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │ ML Model        │  CNN, RNN, Transformer, or ensemble     │
│  │ Inference       │                                         │
│  └────────┬────────┘                                         │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │ Risk Score /    │  Probability, severity, confidence      │
│  │ Classification  │                                         │
│  └─────────────────┘                                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Ethical Considerations

### Privacy Concerns
- Voice is biometric data (personally identifiable)
- Potential for surveillance/monitoring without consent
- Data storage and security requirements
- HIPAA compliance for health applications

### Consent & Transparency
- Clear disclosure when voice is being analyzed
- Opt-in only for health assessments
- Right to access/delete voice data
- Explanation of what's being detected

### Bias & Fairness
- Models may perform differently across demographics
- Training data representation matters
- Age, gender, accent, language considerations
- Validation across diverse populations

### Clinical Validity
- No FDA-approved diagnostics yet
- Should complement, not replace, clinical assessment
- Risk of false positives/negatives
- Appropriate confidence intervals

## Voice Soundboard Implementation

### Biomarker Analysis Module (v1.2.0+)

```python
from voice_soundboard.biomarkers import VoiceBiomarkerAnalyzer

# Initialize analyzer
analyzer = VoiceBiomarkerAnalyzer()

# Analyze voice sample
results = analyzer.analyze("voice_sample.wav")

print(results)
# {
#     "voice_quality": {
#         "jitter_percent": 0.8,
#         "shimmer_percent": 3.2,
#         "hnr_db": 18.5,
#         "cpp_db": 9.2
#     },
#     "prosody": {
#         "f0_mean_hz": 125.3,
#         "f0_std_hz": 28.4,
#         "speaking_rate_sps": 4.2,
#         "pause_ratio": 0.18
#     },
#     "quality_assessment": {
#         "overall_quality": "normal",
#         "breathiness": "low",
#         "roughness": "low",
#         "strain": "none"
#     },
#     "warnings": []  # Any detected concerns
# }
```

### Accessibility Features

```python
# Voice health monitoring for content creators
from voice_soundboard.biomarkers import VoiceHealthMonitor

monitor = VoiceHealthMonitor()

# Track voice quality over time
monitor.add_sample("session_1.wav", timestamp="2026-01-23T10:00:00")
monitor.add_sample("session_2.wav", timestamp="2026-01-23T14:00:00")

# Get fatigue assessment
fatigue = monitor.assess_vocal_fatigue()
print(fatigue)
# {
#     "fatigue_level": "moderate",
#     "recommendation": "Consider a 30-minute voice rest",
#     "trend": "increasing strain over last 4 hours"
# }
```

## Important Disclaimers

```
⚠️ MEDICAL DISCLAIMER

Voice biomarker analysis is for informational and research purposes only.

- NOT a substitute for professional medical diagnosis
- NOT FDA-approved for clinical use
- Results should NOT be used for self-diagnosis
- Always consult healthcare providers for medical concerns

Voice Soundboard provides voice quality metrics for:
- Content creation optimization
- Voice health awareness
- Accessibility features
- Research applications
```

## References

- Vocal Biomarker Trends 2026: https://canaryspeech.com/blog/5-trends-in-2026/
- Voice for Health (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC8138221/
- AI Voice Biomarker for Depression: https://pubmed.ncbi.nlm.nih.gov/39805690/
- Voice as Biomarker for Vocal Fold Lesions: https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1609811/full
- Bridging AI and Healthcare for Voice Biomarkers: https://pmc.ncbi.nlm.nih.gov/articles/PMC12267164/
