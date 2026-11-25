import { useState } from 'react'
import './App.css'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const metrics = [
  { name: 'Audio', mae: 6.2130, rmse: 8.7096 },
  { name: 'Video', mae: 6.0664, rmse: 8.4545 },
  { name: 'Text', mae: 5.1545, rmse: 6.7460 },
  { name: 'Audio + Video', mae: 5.7449, rmse: 7.8202 },
  { name: 'Audio + Text', mae: 5.1575, rmse: 6.3590 },
  { name: 'Video + Text', mae: 4.9956, rmse: 6.3989 },
  { name: 'All Modalities', mae: 5.2522, rmse: 6.2517 },
]

function App() {
  const [page, setPage] = useState('dashboard')

  const [audio, setAudio] = useState(null)
  const [video, setVideo] = useState(null)
  const [transcript, setTranscript] = useState(null)

  const [report, setReport] = useState(null)
  const [screening, setScreening] = useState(null)

  const [loadingQuality, setLoadingQuality] = useState(false)
  const [loadingScreening, setLoadingScreening] = useState(false)
  const [error, setError] = useState('')

  async function handleAssessment() {
    if (!audio && !video && !transcript) {
      setError('Upload at least one file for input quality assessment.')
      return
    }

    setLoadingQuality(true)
    setError('')
    setReport(null)
    setScreening(null)

    const formData = new FormData()

    if (audio) formData.append('audio', audio)
    if (video) formData.append('video', video)
    if (transcript) formData.append('transcript', transcript)

    try {
      const response = await fetch(`${API}/quality/assess`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Assessment failed')
      }

      setReport(data)
    } catch (err) {
      setError(err.message || 'Unable to connect to MindCare API.')
    } finally {
      setLoadingQuality(false)
    }
  }

  async function handleScreening() {
    if (!audio && !video && !transcript) {
      setError('Upload at least one modality for screening.')
      return
    }

    setLoadingScreening(true)
    setError('')
    setScreening(null)
    setReport(null)

    const formData = new FormData()

    if (audio) formData.append('audio', audio)
    if (video) formData.append('video', video)
    if (transcript) formData.append('transcript', transcript)

    formData.append('patient_id', 'Anonymous')

    try {
      const response = await fetch(`${API}/quality/predict`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Screening failed')
      }

      setScreening(data)
    } catch (err) {
      setError(err.message || 'Unable to run screening.')
    } finally {
      setLoadingScreening(false)
    }
  }

  function resetAssessment() {
    setAudio(null)
    setVideo(null)
    setTranscript(null)
    setReport(null)
    setScreening(null)
    setError('')
  }

  function goToAssessment() {
    resetAssessment()
    setPage('assessment')
  }

  function FileInput({ title, accept, file, onChange, description }) {
    return (
      <label className="upload-box">
        <div className="upload-symbol">↑</div>
        <div className="upload-title">{title}</div>
        <div className="upload-description">{description}</div>

        <input
          type="file"
          accept={accept}
          onChange={(e) => onChange(e.target.files?.[0] || null)}
        />

        <div className="upload-selected">
          {file ? `✓ ${file.name}` : 'Click to browse files'}
        </div>
      </label>
    )
  }

  function Dashboard() {
    return (
      <>
        <section className="hero">
          <div className="hero-content">
            <div className="eyebrow">MULTIMODAL RESEARCH SYSTEM</div>

            <h2>
              Understand signals.
              <br />
              Explore patterns.
            </h2>

            <p>
              MindCare is an experimental AI-based screening prototype
              that combines speech, facial features and text to estimate
              PHQ-8 symptom scores.
            </p>

            <button
              className="primary-button hero-button"
              onClick={goToAssessment}
            >
              Start New Assessment <span>→</span>
            </button>
          </div>

          <div className="hero-art">
            <div className="orb orb-one"></div>
            <div className="orb orb-two"></div>
            <div className="orb orb-three"></div>
            <div className="art-center">M</div>
          </div>
        </section>

        <div className="section-heading">
          <div>
            <h3>System Overview</h3>
            <p>What MindCare analyzes</p>
          </div>
        </div>

        <section className="overview-grid">
          <div className="overview-card">
            <div className="overview-icon audio-icon">♫</div>
            <h3>Speech</h3>
            <p>
              Extracts acoustic characteristics from uploaded audio,
              including signal and spectral patterns.
            </p>
            <span className="overview-tag">Audio modality</span>
          </div>

          <div className="overview-card">
            <div className="overview-icon facial-icon">◉</div>
            <h3>Video / Facial Analysis</h3>
            <p>
              Uses video-derived facial measurements, including OpenFace
              action units and landmark features.
            </p>
            <span className="overview-tag">Video modality</span>
          </div>

          <div className="overview-card">
            <div className="overview-icon text-icon">T</div>
            <h3>Text</h3>
            <p>
              Processes transcript text using text vectorization
              and linguistic representation.
            </p>
            <span className="overview-tag">Text modality</span>
          </div>
        </section>

        <section className="info-grid">
          <div className="info-panel">
            <div className="info-panel-icon">◎</div>
            <h3>What is PHQ-8?</h3>
            <p>
              The PHQ-8 is an eight-item questionnaire used to measure
              depressive symptom severity. Its score ranges from 0 to 24.
              MindCare estimates this score from input data instead of
              administering the questionnaire directly.
            </p>
          </div>

          <div className="info-panel">
            <div className="info-panel-icon">⌘</div>
            <h3>How it works</h3>
            <p>
              Upload supported files → extract modality features →
              combine available inputs → generate a PHQ-8 estimate →
              review the research interpretation.
            </p>
          </div>
        </section>

        <section className="disclaimer">
          <h3>Research Disclaimer</h3>
          <p>
            MindCare is an experimental research prototype. Model outputs
            are estimates, not diagnoses. The system is not clinically
            validated for diagnosis or treatment decisions.
          </p>
        </section>
      </>
    )
  }

  function Assessment() {
    return (
      <>
        <section className="assessment-intro">
          <h2>New Assessment</h2>
          <p>
            Upload one or more supported modalities to run an experimental
            screening and input quality analysis.
          </p>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h3>Upload Files</h3>
              <p>Patient identification is not required.</p>
            </div>

            <span className="anonymous-tag">Anonymous</span>
          </div>

          <div className="upload-grid">
            <FileInput
              title="Audio recording"
              accept=".wav"
              file={audio}
              onChange={setAudio}
              description="WAV audio file"
            />

            <FileInput
              title="Video recording"
              accept=".mp4,.avi,.mov"
              file={video}
              onChange={setVideo}
              description="MP4, AVI or MOV video"
            />

            <FileInput
              title="Transcript"
              accept=".csv,.txt,.tsv"
              file={transcript}
              onChange={setTranscript}
              description="CSV, TXT or TSV"
            />
          </div>

          <div className="selected-summary">
            <h4>Selected Inputs</h4>

            <div className="selected-tags">
              {audio && <span className="selected-tag">Audio</span>}
              {video && <span className="selected-tag">Video</span>}
              {transcript && <span className="selected-tag">Text</span>}

              {!audio && !video && !transcript && (
                <span className="empty-selection">
                  No files selected
                </span>
              )}
            </div>

            <p>
              Screening and input quality analysis use the uploaded audio,
              video and transcript. Facial measurements are extracted
              automatically from the video.
            </p>
          </div>

          {error && <div className="error-message">{error}</div>}

          <div className="assessment-actions">
            <button
              className="primary-button"
              onClick={handleScreening}
              disabled={loadingScreening}
            >
              {loadingScreening ? 'Running Model...' : 'Run Screening'}
            </button>

            <button
              className="secondary-button"
              onClick={handleAssessment}
              disabled={loadingQuality}
            >
              {loadingQuality ? 'Assessing...' : 'Analyze Input Quality'}
            </button>

            <button className="text-button" onClick={resetAssessment}>
              Clear Files
            </button>
          </div>
        </section>

        {screening && (
          <section className="panel report-panel">
            <div className="panel-header">
              <div>
                <h3>Screening Report</h3>
                <p>Experimental model output</p>
              </div>

              <span className="anonymous-tag">Anonymous</span>
            </div>

            <div className="report-details">
              <h4>Uploaded Modalities</h4>
              <p>
                {screening.uploaded_modalities?.map((m) =>
                  m.charAt(0).toUpperCase() + m.slice(1)
                ).join(' + ') || 'Not reported'}
              </p>
            </div>

            <div className="overall-score">
              <span>Estimated PHQ-8 Score</span>

              <h2>
                {Number(screening.estimated_phq8_score).toFixed(2)} / 24
              </h2>

              <p>{screening.estimated_range}</p>
            </div>

            <div className="score-scale">
              <div className="scale-bar">
                <span className="scale-minimal"></span>
                <span className="scale-mild"></span>
                <span className="scale-moderate"></span>
                <span className="scale-severe"></span>
              </div>

              <div className="scale-labels">
                <span>Minimal</span>
                <span>Mild</span>
                <span>Moderate</span>
                <span>Severe</span>
              </div>
            </div>

            <div className="report-details">
              <h4>Interpretation</h4>

              <p>
                {Number(screening.estimated_phq8_score) < 5
                  ? 'The estimated score falls within the minimal symptom range.'
                  : Number(screening.estimated_phq8_score) < 10
                  ? 'The estimated score falls within the mild symptom range.'
                  : Number(screening.estimated_phq8_score) < 15
                  ? 'The estimated score falls within the moderate symptom range.'
                  : Number(screening.estimated_phq8_score) < 20
                  ? 'The estimated score falls within the moderately severe symptom range.'
                  : 'The estimated score falls within the severe symptom range.'}
              </p>

              <p>
                This is a model-generated estimate, not a diagnosis.
                A questionnaire score alone cannot determine whether
                someone has depression.
              </p>

              <h4>Model</h4>
              <p>{screening.model}</p>

              <h4>Processing Device</h4>
              <p>{screening.device}</p>

              <p className="report-note">{screening.disclaimer}</p>
            </div>

            <button className="secondary-button" onClick={resetAssessment}>
              New Assessment
            </button>
          </section>
        )}

        {report && (
          <section className="panel report-panel">
            <div className="panel-header">
              <div>
                <h3>Input Quality Report</h3>
                <p>Technical assessment of uploaded inputs</p>
              </div>
            </div>

            <div className="overall-score">
              <span>Overall Quality Score</span>

              <h2>
                {report.overall_quality_score ?? 'N/A'}
                {report.overall_quality_score !== null ? '/100' : ''}
              </h2>

              <p>{report.overall_status}</p>
            </div>

            <div className="report-grid">
              {['audio', 'video', 'text'].map((modality) => (
                <div className="report-card" key={modality}>
                  <h4>
                    {modality.charAt(0).toUpperCase() + modality.slice(1)}
                  </h4>

                  <p>
                    Status:{' '}
                    {report[modality]?.available
                      ? 'Available'
                      : 'Not available'}
                  </p>

                  {report.quality_scores?.[modality] !== undefined && (
                    <p>
                      Quality score: {report.quality_scores[modality]}/100
                    </p>
                  )}

                  {report[modality]?.warnings?.length > 0 && (
                    <div className="warning-list">
                      <strong>Warnings</strong>

                      {report[modality].warnings.map((warning, i) => (
                        <p key={i}>{warning}</p>
                      ))}
                    </div>
                  )}

                  {report[modality]?.error && (
                    <p className="error-message">
                      {report[modality].error}
                    </p>
                  )}
                </div>
              ))}
            </div>

            <div className="report-details">
              <h4>Uploaded Modalities</h4>

              <p>
                {report.modalities_available?.join(' + ') || 'None'}
              </p>

              {report.overall_warnings?.length > 0 && (
                <>
                  <h4>Overall Warnings</h4>

                  {report.overall_warnings.map((warning, i) => (
                    <p className="warning-text" key={i}>
                      {warning}
                    </p>
                  ))}
                </>
              )}

              <p className="report-note">{report.score_note}</p>
            </div>
          </section>
        )}
      </>
    )
  }

  function AboutModel() {
    return (
      <>
        <section className="assessment-intro">
          <h2>About the Model</h2>
          <p>
            Model architecture, research evaluation and modality comparison
            based on the project's current development evaluation.
          </p>
        </section>

        <section className="model-summary-grid">
          <div className="model-summary-card">
            <span>Architecture</span>
            <h3>Multimodal Neural Network</h3>
            <p>Three modality branches</p>
          </div>

          <div className="model-summary-card">
            <span>Prediction Target</span>
            <h3>PHQ-8 Regression</h3>
            <p>Estimated score · 0–24</p>
          </div>

          <div className="model-summary-card">
            <span>Evaluation Set</span>
            <h3>Development</h3>
            <p>35 participants</p>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h3>Modality Performance</h3>
              <p>Development set · Mean Absolute Error (MAE)</p>
            </div>
          </div>

          <div className="chart">
            {metrics.map((item) => (
              <div className="chart-row" key={item.name}>
                <div className="chart-label">{item.name}</div>

                <div className="chart-track">
                  <div
                    className={`chart-bar ${
                      item.name === 'All Modalities' ? 'highlight-bar' : ''
                    }`}
                    style={{ width: `${(item.mae / 8) * 100}%` }}
                  ></div>
                </div>

                <div className="chart-value">
                  {item.mae.toFixed(2)}
                </div>
              </div>
            ))}
          </div>

          <p className="chart-note">
            Lower MAE indicates predictions closer to the reference PHQ-8 scores.
            These are development-set results and are not independent test results.
          </p>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h3>Evaluation Metrics</h3>
              <p>Development-set model comparison</p>
            </div>
          </div>

          <div className="table-wrapper">
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Input</th>
                  <th>MAE</th>
                  <th>RMSE</th>
                </tr>
              </thead>

              <tbody>
                {metrics.map((item) => (
                  <tr
                    key={item.name}
                    className={
                      item.name === 'All Modalities' ? 'table-highlight' : ''
                    }
                  >
                    <td>{item.name}</td>
                    <td>{item.mae.toFixed(4)}</td>
                    <td>{item.rmse.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h3>Model Information</h3>
              <p>Experimental research implementation</p>
            </div>
          </div>

          <div className="model-info-list">
            <div>
              <strong>Speech branch</strong>
              <p>Extracted acoustic feature representation.</p>
            </div>

            <div>
              <strong>Facial branch</strong>
              <p>OpenFace-based action unit and landmark representation.</p>
            </div>

            <div>
              <strong>Text branch</strong>
              <p>TF-IDF transcript feature representation.</p>
            </div>

            <div>
              <strong>Output</strong>
              <p>Regression-based estimated PHQ-8 score.</p>
            </div>

            <div>
              <strong>Evaluation protocol</strong>
              <p>
                Participant-level dataset splits. Development data is used
                for model evaluation during development. The test split
                should remain untouched during model selection.
              </p>
            </div>
          </div>
        </section>

        <section className="disclaimer">
          <h3>Limitations</h3>
          <p>
            Development-set performance is not evidence of clinical
            effectiveness. Model predictions can be inaccurate, and
            modality features are not direct or validated indicators
            of depression. MindCare is not a diagnostic system.
          </p>
        </section>
      </>
    )
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">M</div>

          <div>
            <h2>MindCare</h2>
            <span>AI Screening Prototype</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav>
          <button
            className={`nav-item ${page === 'dashboard' ? 'active' : ''}`}
            onClick={() => setPage('dashboard')}
          >
            <span className="nav-icon">⌂</span> Dashboard
          </button>

          <button
            className={`nav-item ${page === 'assessment' ? 'active' : ''}`}
            onClick={goToAssessment}
          >
            <span className="nav-icon">＋</span> New Assessment
          </button>

          <button
            className={`nav-item ${page === 'model' ? 'active' : ''}`}
            onClick={() => setPage('model')}
          >
            <span className="nav-icon">⌘</span> About Model
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="status-dot"></div>

          <div>
            <strong>MindCare API</strong>
            <small>Local development server</small>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="breadcrumb">
              MindCare /{' '}
              {page === 'dashboard'
                ? 'Dashboard'
                : page === 'assessment'
                ? 'New Assessment'
                : 'About Model'}
            </div>

            <h1>
              {page === 'dashboard'
                ? 'Dashboard'
                : page === 'assessment'
                ? 'New Assessment'
                : 'About Model'}
            </h1>
          </div>

          <div className="profile">M</div>
        </header>

        {page === 'dashboard' && <Dashboard />}
        {page === 'assessment' && <Assessment />}
        {page === 'model' && <AboutModel />}
      </main>
    </div>
  )
}

export default App