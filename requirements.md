# Requirements Document: Sparkle Labs Early Screening & Clinical Decision Support System

## Project Overview

**Project Name:** Sparkle Labs - Learn. Grow. Sparkle.  
**System Name:** Aster AI Adventure Guide  
**Purpose:** Early screening and clinical decision support for developmental assessment through gamified, privacy-preserving behavioral analysis

## Scope & Scientific Positioning

### Core Positioning
- **Primary Function:** Early screening & clinical decision support only
- **Explicit Non-Goal:** Not a diagnostic system (no ASD vs non-ASD classification)
- **Rationale:** Lack of large, publicly available, clinically validated diagnostic datasets
- **Approach:** System quantifies task-specific behavioral deviation, aligned with screening practice and early-intervention research

### Universal Positioning Strategy
- **Public Face:** A gamified development platform for all children that removes clinical stigma
- **Clinical Value:** Provides professionals with objective AI biomarkers to replace subjective, delayed diagnostic methods
- **Data Strategy:** Building a massive data moat for early intervention through universal engagement

## System Architecture

### Phase 1: Offline Training & Calibration (Models)

#### Vision Processing (Task-Relative Baseline)
- **Dataset:** DREAM (structured imitation & joint-attention tasks)
- **Model:** Task-specific TCN-VAE (offline calibration)
- **Signals:** 
  - Gaze tracking and latency
  - Head pose dynamics
  - Gesture timing
  - Motor smoothness
  - Joint attention patterns
- **Learning Objectives:** Expected task dynamics & tolerance bands
- **Outputs:** Task-specific ranges (p90/p95), metric weights

#### Audio Processing (Normative-Temporal Baseline)
- **Datasets:** CHILDES (TD norms) + ASDBank AAC
- **Model:** TCN-AE / TCN-VAE (offline calibration)
- **Features/Signals/Biomarkers:**
  - Prosody analysis
  - Voice quality metrics
  - Conversational dynamics (MLT Ratio)
  - Response latency
- **Outputs:** Timing ranges, tolerance bands, metric weights

#### EEG Processing (Optional Clinical Add-on)
- **Dataset:** TUH/DEAP priors
- **Model:** TCN-VAE
- **Signals:**
  - θ/α ratios
  - β/α ratios
  - ERP latency measurements

### Phase 2: Runtime Screening System

#### Guided Screening Tasks (Client-Side)
- **Task Framework:** Child follows pre-recorded standardized videos (DREAM-aligned tasks)
- **Capture Modalities:**
  - RGB video
  - Audio
  - Optional EEG (clinical plug-in)
  - Task metadata
- **Privacy Requirement:** Zero raw footage storage

#### Client-Side Feature Extraction (Privacy-Preserving)
- **Implementation:** Fully in-browser processing (MediaPipe, Web Audio API)
- **Vision Features:**
  - Gaze latency
  - Head-turn dynamics
  - Gesture delay
  - Motor smoothness
- **Audio Features:**
  - Vocalization rate
  - Response latency
  - Prosodic variability
- **EEG Features (Optional):**
  - θ/α ratios
  - β/α ratios
  - ERP latency

#### Deviation Estimation (FastAPI Backend)
- **Processing Approach:** Task-relative (vision) + norm-relative (audio), weighted aggregation
- **Architecture:** Lightweight, interpretable, privacy-preserving runtime processing

#### Temporal Surprise & Risk Output
- **Surprise Curves:** Deviation vs time (vision + audio combined)
- **Breakdown Detection:** Pinpoints when interaction breakdown occurs
- **Screening Risk Index:** 90th percentile of task deviation to explain AI flagging for clinical review
- **Output Bands:** Low / Moderate / High risk categories

#### Confidence Gating System
- **Suppression Triggers:**
  - Low face/pose confidence
  - Occlusion or motion blur
  - Low audio SNR
- **Purpose:** Reduces false positives through quality control

### Phase 3: Clinical Support & Follow-Up

#### Agentic AI Layer
- **Screening Agent:** Task orchestration & scoring
- **Clinical Support Agent:** Interpretable summaries & timelines
- **Therapy Planning Agent:** Personalized activity suggestions (e.g., speech therapy with engaging games and lessons)
- **Progress Monitoring Agent:** Longitudinal tracking

#### Therapy Planning (ESDM-Aligned)
- **Framework:** Grounded in Early Start Denver Model (ESDM)
- **AI Adaptations:**
  - Adaptive task difficulty
  - Personalized reward schedules
  - Focus prioritization (attention, imitation, communication)
  - Delivered via gamified, guided sessions
- **Dynamic Features:** Real-time difficulty adjustment based on continuous feedback

## Technical Requirements

### Performance Metrics
- **Primary Metrics:**
  - Latent deviation (likelihood)
  - Z-score from normative space
  - Sensitivity at fixed specificity (interaction-level)
  - Temporal surprise curves
- **Edge Metrics:**
  - Processing latency
  - Model size optimization

### Scalability Requirements
- **Offline Processing:** Deep temporal models for calibration only
- **Runtime Processing:** Lightweight, interpretable, privacy-preserving
- **Dynamic Care:** Continuous AI-driven progress monitoring and agentic orchestration

### Privacy & Security Requirements
- **Data Handling:** Zero raw footage storage
- **Processing:** Client-side feature extraction only
- **Transmission:** Only processed features and metadata
- **Compliance:** HIPAA-ready architecture for clinical deployment

## User Experience Requirements

### Gamification Elements
- **Character:** Aster as AI adventure guide
- **Engagement:** Interactive, game-like assessment tasks
- **Progression:** Achievement-based advancement system
- **Accessibility:** Age-appropriate interface design

### Clinical Interface
- **Dashboard:** Professional-grade analytics and reporting
- **Interpretability:** Clear explanation of AI flagging decisions
- **Integration:** Compatible with existing clinical workflows
- **Documentation:** Comprehensive audit trails for clinical records

## Development Deliverables

### Core System Components
1. **Client-Side Application:** Browser-based assessment platform
2. **Feature Extraction Pipeline:** Privacy-preserving processing modules
3. **Backend API:** FastAPI-based deviation estimation service
4. **Agentic AI Layer:** Multi-agent clinical support system
5. **Clinical Dashboard:** Professional interface and reporting tools

### Documentation Requirements
- Technical architecture documentation
- Clinical validation protocols
- User training materials
- API documentation
- Privacy and security compliance documentation

## Success Criteria

### Technical Metrics
- Sub-second processing latency for real-time feedback
- >95% uptime for clinical deployment
- WCAG 2.1 AA accessibility compliance
- Zero data breach incidents

### Clinical Metrics
- Sensitivity/specificity benchmarks against clinical gold standards
- Clinician adoption and satisfaction rates
- Reduction in time-to-intervention metrics
- Longitudinal outcome tracking effectiveness

### Business Metrics
- User engagement and retention rates
- Clinical partner adoption
- Data collection volume for model improvement
- Revenue targets for sustainable operation
