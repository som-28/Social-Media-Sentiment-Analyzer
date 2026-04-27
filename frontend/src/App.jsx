import { useEffect, useMemo, useState } from 'react'
import SentimentChart from './components/SentimentChart'
import TrendChart from './components/TrendChart'
import GeoHeatmap from './components/GeoHeatmap'
import TopicClusters from './components/TopicClusters'
import {
  compareModels,
  exportReport,
  fetchLiveSummary,
  fetchTrendSeries,
  fetchTodayDashboard,
  fetchHistoricalTrends,
  fetchDrilldown,
  fetchRecommendations,
  generateExecutiveSummary,
  listSnapshots,
  listWatchlists,
  listReportSchedules,
  listReportDeliveries,
  predictCsv,
  predictText,
  createWatchlist,
  createReportSchedule,
  runWatchlist,
  deleteWatchlist,
  runReportScheduleNow,
  deleteReportSchedule,
  saveSnapshot,
  simulateScenario,
  ingestLiveSource,
} from './api'

const MODELS = [
  { value: 'tfidf_lr', label: 'TF-IDF + Logistic Regression' },
  { value: 'word2vec_svm', label: 'Word2Vec + SVM (proxy)' },
  { value: 'distilbert', label: 'DistilBERT (ensemble proxy)' },
]

export default function App() {
  const [model, setModel] = useState('distilbert')
  const [text, setText] = useState('')
  const [singleResult, setSingleResult] = useState(null)
  const [csvResult, setCsvResult] = useState(null)
  const [loadingText, setLoadingText] = useState(false)
  const [loadingCsv, setLoadingCsv] = useState(false)
  const [csvProgress, setCsvProgress] = useState(0)
  const [liveSummary, setLiveSummary] = useState(null)
  const [trendSeries, setTrendSeries] = useState([])
  const [loadingLive, setLoadingLive] = useState(true)
  const [liveEnabled, setLiveEnabled] = useState(true)
  const [todayDashboard, setTodayDashboard] = useState(null)
  const [recommendations, setRecommendations] = useState([])
  const [historicalTrends, setHistoricalTrends] = useState([])
  const [drilldownResults, setDrilldownResults] = useState([])
  const [drilldownLabel, setDrilldownLabel] = useState('')
  const [drilldownTopic, setDrilldownTopic] = useState('')
  const [loadingDrilldown, setLoadingDrilldown] = useState(false)
  const [comparing, setComparing] = useState(false)
  const [compareData, setCompareData] = useState(null)
  const [scenarioData, setScenarioData] = useState(null)
  const [executiveSummary, setExecutiveSummary] = useState('')
  const [snapshots, setSnapshots] = useState([])
  const [snapshotRole, setSnapshotRole] = useState('analyst')
  const [reportPreview, setReportPreview] = useState('')
  const [reportDownloadUrl, setReportDownloadUrl] = useState('')
  const [ingestingLive, setIngestingLive] = useState(false)
  const [sourceType, setSourceType] = useState('manual')
  const [sourceUrl, setSourceUrl] = useState('')
  const [sourceLimit, setSourceLimit] = useState(30)
  const [sourceTextField, setSourceTextField] = useState('text')
  const [manualTextsRaw, setManualTextsRaw] = useState('')
  const [watchlistName, setWatchlistName] = useState('')
  const [watchlists, setWatchlists] = useState([])
  const [scheduleDestination, setScheduleDestination] = useState('team@example.com')
  const [scheduleChannel, setScheduleChannel] = useState('email')
  const [scheduleInterval, setScheduleInterval] = useState(60)
  const [reportSchedules, setReportSchedules] = useState([])
  const [reportDeliveries, setReportDeliveries] = useState([])
  const [error, setError] = useState('')

  const chartDistribution = useMemo(() => {
    if (csvResult?.distribution) return csvResult.distribution
    if (singleResult?.label) return { [singleResult.label]: 1 }
    return {}
  }, [csvResult, singleResult])

  const csvInsight = useMemo(() => {
    if (!csvResult?.distribution || !csvResult?.count) return null

    const positive = Number(csvResult.distribution.positive || 0)
    const neutral = Number(csvResult.distribution.neutral || 0)
    const negative = Number(csvResult.distribution.negative || 0)
    const total = Number(csvResult.count || positive + neutral + negative)

    if (!total) return null

    const stats = [
      { label: 'positive', value: positive },
      { label: 'neutral', value: neutral },
      { label: 'negative', value: negative },
    ]
    const top = stats.reduce((best, current) => (current.value > best.value ? current : best), stats[0])
    const topPct = Math.round((top.value / total) * 100)

    let message = `Most posts are ${top.label} (${topPct}%).`
    if (top.label === 'positive') {
      message = `Overall mood is mostly positive (${topPct}%), which suggests favorable audience reactions.`
    }
    if (top.label === 'neutral') {
      message = `Overall mood is mostly neutral (${topPct}%), which suggests people are discussing rather than reacting strongly.`
    }
    if (top.label === 'negative') {
      message = `Overall mood is mostly negative (${topPct}%), which may indicate dissatisfaction or concern in your audience.`
    }

    return {
      message,
      positivePct: Math.round((positive / total) * 100),
      neutralPct: Math.round((neutral / total) * 100),
      negativePct: Math.round((negative / total) * 100),
    }
  }, [csvResult])

  const csvEmotionInsight = useMemo(() => {
    if (!csvResult?.emotion_distribution || !csvResult?.count) return null

    const entries = Object.entries(csvResult.emotion_distribution)
      .map(([label, value]) => ({ label, value: Number(value || 0) }))
      .filter((entry) => entry.value > 0)

    if (!entries.length) return null
    const total = entries.reduce((sum, entry) => sum + entry.value, 0)
    const top = entries.reduce((best, current) => (current.value > best.value ? current : best), entries[0])
    const pct = Math.round((top.value / total) * 100)

    return `Dominant emotion: ${top.label} (${pct}% of analyzed rows).`
  }, [csvResult])

  const singleInsight = useMemo(() => {
    if (!singleResult?.label) return null

    const confidencePct = Math.round(Number(singleResult.confidence || 0) * 100)
    if (singleResult.label === 'positive') {
      return `This text reads as positive. The model is about ${confidencePct}% confident.`
    }
    if (singleResult.label === 'negative') {
      return `This text reads as negative. The model is about ${confidencePct}% confident.`
    }
    return `This text reads as neutral. The model is about ${confidencePct}% confident.`
  }, [singleResult])

  const singleEmotionInsight = useMemo(() => {
    if (!singleResult?.emotion?.label) return null
    const confidence = Math.round(Number(singleResult.emotion.confidence || 0) * 100)
    return `Detected emotion: ${singleResult.emotion.label} (${confidence}% confidence).`
  }, [singleResult])

  const sarcasmInsight = useMemo(() => {
    if (!singleResult?.sarcasm) return null
    const pct = Math.round(Number(singleResult.sarcasm.score || 0) * 100)
    if (singleResult.sarcasm.detected) {
      return `Possible sarcasm detected (${pct}% confidence): ${singleResult.sarcasm.reason}.`
    }
    return `Sarcasm risk is low (${pct}% confidence).`
  }, [singleResult])

  const toxicityInsight = useMemo(() => {
    if (!singleResult?.toxicity) return null
    const pct = Math.round(Number(singleResult.toxicity.score || 0) * 100)
    if (singleResult.toxicity.level === 'none') {
      return `Toxicity level: none (${pct}%).`
    }
    const terms = Array.isArray(singleResult.toxicity.terms) ? singleResult.toxicity.terms.join(', ') : ''
    return `Toxicity level: ${singleResult.toxicity.level} (${pct}%). ${terms ? `Matched terms: ${terms}.` : ''}`
  }, [singleResult])

  const liveInsight = useMemo(() => {
    if (!liveSummary?.distribution || !liveSummary.window_size) return null

    const positive = Number(liveSummary.distribution.positive || 0)
    const neutral = Number(liveSummary.distribution.neutral || 0)
    const negative = Number(liveSummary.distribution.negative || 0)
    const total = positive + neutral + negative
    if (!total) return 'No live predictions yet. Run text or CSV analysis to start the stream.'

    const stats = [
      { label: 'positive', value: positive },
      { label: 'neutral', value: neutral },
      { label: 'negative', value: negative },
    ]
    const top = stats.reduce((best, current) => (current.value > best.value ? current : best), stats[0])
    const pct = Math.round((top.value / total) * 100)

    if (top.label === 'positive') {
      return `Live trend: positive tone leads (${pct}% of recent predictions).`
    }
    if (top.label === 'negative') {
      return `Live trend: negative tone leads (${pct}% of recent predictions).`
    }
    return `Live trend: neutral tone leads (${pct}% of recent predictions).`
  }, [liveSummary])

  useEffect(() => {
    let active = true

    async function refreshLiveSummary() {
      try {
        if (active) setLoadingLive(true)
        const [data, trend, dashboardData, historicalData, recommendationData] = await Promise.all([
          fetchLiveSummary({ window: 100 }),
          fetchTrendSeries({ window: 150 }),
          fetchTodayDashboard({ window: 200 }),
          fetchHistoricalTrends({ bucket: 'day', window: 1000 }),
          fetchRecommendations({ window: 200 }),
        ])
        if (active) {
          setLiveSummary(data)
          setTrendSeries(Array.isArray(trend.points) ? trend.points : [])
          setTodayDashboard(dashboardData)
          setHistoricalTrends(Array.isArray(historicalData.points) ? historicalData.points : [])
          setRecommendations(Array.isArray(recommendationData.recommendations) ? recommendationData.recommendations : [])
        }
      } catch {
        if (active) {
          setLiveSummary(null)
          setTrendSeries([])
          setTodayDashboard(null)
          setHistoricalTrends([])
          setRecommendations([])
        }
      } finally {
        if (active) setLoadingLive(false)
      }
    }

    refreshLiveSummary()
    if (!liveEnabled) {
      return () => {
        active = false
      }
    }

    const interval = setInterval(refreshLiveSummary, 5000)
    return () => {
      active = false
      clearInterval(interval)
    }
  }, [liveEnabled])

  useEffect(() => {
    if (!loadingCsv) {
      return undefined
    }

    setCsvProgress(8)
    const timer = setInterval(() => {
      setCsvProgress((current) => {
        if (current >= 92) {
          return current
        }
        const step = current < 40 ? 8 : current < 70 ? 5 : 2
        return Math.min(92, current + step)
      })
    }, 350)

    return () => {
      clearInterval(timer)
    }
  }, [loadingCsv])

  async function loadDrilldown({ label = drilldownLabel, topic = drilldownTopic } = {}) {
    setLoadingDrilldown(true)
    setError('')
    try {
      const data = await fetchDrilldown({ label, topic, limit: 40 })
      setDrilldownResults(Array.isArray(data.results) ? data.results : [])
    } catch (err) {
      setError(err.message)
      setDrilldownResults([])
    } finally {
      setLoadingDrilldown(false)
    }
  }

  async function onSentimentBarSelect(label) {
    setDrilldownLabel(label)
    await loadDrilldown({ label, topic: drilldownTopic })
  }

  async function onSubmitText(event) {
    event.preventDefault()
    setError('')
    setLoadingText(true)
    setCsvResult(null)

    try {
      const data = await predictText({ text, model })
      setSingleResult(data)
      setCompareData(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoadingText(false)
    }
  }

  async function onUploadCsv(event) {
    const file = event.target.files?.[0]
    if (!file) return

    setError('')
    setLoadingCsv(true)
    setCsvProgress(5)
    setSingleResult(null)

    try {
      const data = await predictCsv({ file, model })
      setCsvResult(data)
      setExecutiveSummary('')
      setScenarioData(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setCsvProgress(100)
      setLoadingCsv(false)
      event.target.value = ''
      setTimeout(() => setCsvProgress(0), 700)
    }
  }

  async function onCompareModels() {
    if (!text.trim()) return
    setError('')
    setComparing(true)
    try {
      const data = await compareModels({ text })
      setCompareData(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setComparing(false)
    }
  }

  async function onGenerateSummary() {
    if (!csvResult?.distribution) return
    setError('')
    try {
      const data = await generateExecutiveSummary({ distribution: csvResult.distribution })
      setExecutiveSummary(String(data.summary || ''))
    } catch (err) {
      setError(err.message)
    }
  }

  async function onSimulateScenario() {
    if (!csvResult?.distribution) return
    setError('')
    try {
      const data = await simulateScenario({
        distribution: csvResult.distribution,
        deltaNegativePct: 15,
      })
      setScenarioData(data)
    } catch (err) {
      setError(err.message)
    }
  }

  async function onSaveSnapshot() {
    const payload = {
      singleResult,
      csvResult,
      compareData,
      executiveSummary,
      scenarioData,
    }
    try {
      await saveSnapshot({ title: `Snapshot ${new Date().toLocaleString()}`, role: snapshotRole, data: payload })
      const refreshed = await listSnapshots()
      setSnapshots(Array.isArray(refreshed.snapshots) ? refreshed.snapshots : [])
    } catch (err) {
      setError(err.message)
    }
  }

  async function onLoadSnapshots() {
    try {
      const refreshed = await listSnapshots()
      setSnapshots(Array.isArray(refreshed.snapshots) ? refreshed.snapshots : [])
    } catch (err) {
      setError(err.message)
    }
  }

  async function onExportReport() {
    if (!csvResult?.distribution) return
    try {
      const data = await exportReport({
        title: 'Social Media Sentiment Report',
        distribution: csvResult.distribution,
        topics: csvResult.topics || [],
      })
      if (data.format === 'pdf-base64' && data.content_base64) {
        const link = `data:application/pdf;base64,${data.content_base64}`
        setReportDownloadUrl(link)
        setReportPreview(`PDF generated (${data.size_bytes || 0} bytes). Use download link below.`)
      } else {
        setReportPreview(String(data.content || ''))
      }
    } catch (err) {
      setError(err.message)
    }
  }

  async function onIngestLiveSource(event) {
    event.preventDefault()
    setError('')
    setIngestingLive(true)
    try {
      const manualTexts = manualTextsRaw
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean)
      const data = await ingestLiveSource({
        sourceType,
        sourceUrl,
        model,
        limit: Number(sourceLimit || 30),
        textField: sourceTextField,
        manualTexts,
      })
      setCsvResult(data)
      setSingleResult(null)
      setCompareData(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setIngestingLive(false)
    }
  }

  async function onSaveWatchlist() {
    setError('')
    try {
      const manualTexts = manualTextsRaw
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean)
      await createWatchlist({
        name: watchlistName || `Watchlist ${new Date().toLocaleString()}`,
        sourceType,
        sourceUrl,
        model,
        limit: Number(sourceLimit || 30),
        textField: sourceTextField,
        manualTexts,
      })
      setWatchlistName('')
      const fresh = await listWatchlists()
      setWatchlists(Array.isArray(fresh.watchlists) ? fresh.watchlists : [])
    } catch (err) {
      setError(err.message)
    }
  }

  async function onLoadWatchlists() {
    setError('')
    try {
      const fresh = await listWatchlists()
      setWatchlists(Array.isArray(fresh.watchlists) ? fresh.watchlists : [])
    } catch (err) {
      setError(err.message)
    }
  }

  async function onRunWatchlist(id) {
    setError('')
    try {
      const data = await runWatchlist(id)
      setCsvResult(data)
      setSingleResult(null)
      setCompareData(null)
    } catch (err) {
      setError(err.message)
    }
  }

  async function onDeleteWatchlist(id) {
    setError('')
    try {
      await deleteWatchlist(id)
      const fresh = await listWatchlists()
      setWatchlists(Array.isArray(fresh.watchlists) ? fresh.watchlists : [])
    } catch (err) {
      setError(err.message)
    }
  }

  async function onCreateSchedule() {
    if (!csvResult?.distribution) {
      setError('Run CSV or live ingestion first to schedule reports.')
      return
    }
    setError('')
    try {
      await createReportSchedule({
        title: 'Automated Sentiment Report',
        destination: scheduleDestination,
        channel: scheduleChannel,
        intervalMinutes: Number(scheduleInterval || 60),
        distribution: csvResult.distribution,
        topics: csvResult.topics || [],
      })
      const fresh = await listReportSchedules()
      setReportSchedules(Array.isArray(fresh.schedules) ? fresh.schedules : [])
    } catch (err) {
      setError(err.message)
    }
  }

  async function onLoadSchedules() {
    setError('')
    try {
      const [scheduleData, deliveryData] = await Promise.all([
        listReportSchedules(),
        listReportDeliveries(),
      ])
      setReportSchedules(Array.isArray(scheduleData.schedules) ? scheduleData.schedules : [])
      setReportDeliveries(Array.isArray(deliveryData.deliveries) ? deliveryData.deliveries : [])
    } catch (err) {
      setError(err.message)
    }
  }

  async function onRunScheduleNow(id) {
    setError('')
    try {
      await runReportScheduleNow(id)
      const deliveryData = await listReportDeliveries()
      setReportDeliveries(Array.isArray(deliveryData.deliveries) ? deliveryData.deliveries : [])
    } catch (err) {
      setError(err.message)
    }
  }

  async function onDeleteSchedule(id) {
    setError('')
    try {
      await deleteReportSchedule(id)
      const fresh = await listReportSchedules()
      setReportSchedules(Array.isArray(fresh.schedules) ? fresh.schedules : [])
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <h1>Social Media Sentiment Analyzer</h1>
        <p>
          Analyze social posts using three model strategies and view confidence + distribution in real time.
        </p>
      </header>

      <main className="layout">
        <section className="panel">
          <h2>Input</h2>

          <label className="field">
            <span>Select Model</span>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {MODELS.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>

          <form onSubmit={onSubmitText} className="field">
            <span>Single Text Prediction</span>
            <textarea
              rows={6}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type a tweet, review, or post text..."
              required
            />
            <button disabled={loadingText}>{loadingText ? 'Analyzing...' : 'Analyze Text'}</button>
            <button type="button" className="secondary-btn" disabled={comparing} onClick={onCompareModels}>
              {comparing ? 'Comparing...' : 'Compare All Models'}
            </button>
          </form>

          <label className="field upload">
            <span>Batch CSV Prediction</span>
            <p>
              Upload a CSV with a post column like text, tweet, content, message, or review. Add location/city/country to unlock geo insights.
            </p>
            <input type="file" accept=".csv" onChange={onUploadCsv} disabled={loadingCsv} />
            {loadingCsv && (
              <div className="progress-wrap" role="status" aria-live="polite">
                <small>Processing CSV... {csvProgress}%</small>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${csvProgress}%` }} />
                </div>
              </div>
            )}
          </label>

          <form className="field" onSubmit={onIngestLiveSource}>
            <span>Live Source Ingestion</span>
            <select value={sourceType} onChange={(event) => setSourceType(event.target.value)}>
              <option value="manual">Manual Lines</option>
              <option value="reddit_json">Reddit JSON URL</option>
              <option value="rss">RSS Feed URL</option>
              <option value="json_feed">Generic JSON Feed URL</option>
            </select>
            {sourceType === 'manual' ? (
              <textarea
                rows={4}
                value={manualTextsRaw}
                onChange={(event) => setManualTextsRaw(event.target.value)}
                placeholder="Enter one post per line for quick ingestion..."
              />
            ) : (
              <input
                type="url"
                value={sourceUrl}
                onChange={(event) => setSourceUrl(event.target.value)}
                placeholder="https://example.com/feed"
                required={sourceType !== 'manual'}
              />
            )}
            <div className="inline-grid">
              <label>
                Limit
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={sourceLimit}
                  onChange={(event) => setSourceLimit(event.target.value)}
                />
              </label>
              <label>
                Text Field (JSON)
                <input
                  type="text"
                  value={sourceTextField}
                  onChange={(event) => setSourceTextField(event.target.value)}
                  placeholder="text"
                />
              </label>
            </div>
            <button type="submit" className="secondary-btn" disabled={ingestingLive}>
              {ingestingLive ? 'Ingesting...' : 'Ingest And Analyze Live Source'}
            </button>
            <div className="button-row">
              <input
                type="text"
                value={watchlistName}
                onChange={(event) => setWatchlistName(event.target.value)}
                placeholder="Optional watchlist name"
              />
              <button type="button" className="secondary-btn" onClick={onSaveWatchlist}>
                Save Watchlist
              </button>
              <button type="button" className="secondary-btn" onClick={onLoadWatchlists}>
                Load Watchlists
              </button>
            </div>
          </form>

          {watchlists.length > 0 && (
            <article className="result stream">
              <h3>Saved Watchlists</h3>
              <ul>
                {watchlists.slice(0, 8).map((item) => (
                  <li key={item.id}>
                    <strong>{item.name}</strong> ({item.source_type})
                    <div className="button-row">
                      <button type="button" className="secondary-btn" onClick={() => onRunWatchlist(item.id)}>
                        Run
                      </button>
                      <button type="button" className="secondary-btn" onClick={() => onDeleteWatchlist(item.id)}>
                        Delete
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            </article>
          )}

          <article className="result stream">
            <h3>Scheduled Report Delivery</h3>
            <div className="inline-grid">
              <label>
                Destination
                <input
                  type="text"
                  value={scheduleDestination}
                  onChange={(event) => setScheduleDestination(event.target.value)}
                />
              </label>
              <label>
                Channel
                <select value={scheduleChannel} onChange={(event) => setScheduleChannel(event.target.value)}>
                  <option value="email">email</option>
                  <option value="slack">slack</option>
                  <option value="teams">teams</option>
                </select>
              </label>
              <label>
                Interval (minutes)
                <input
                  type="number"
                  min={1}
                  max={10080}
                  value={scheduleInterval}
                  onChange={(event) => setScheduleInterval(event.target.value)}
                />
              </label>
            </div>
            <div className="button-row">
              <button type="button" className="secondary-btn" onClick={onCreateSchedule}>
                Create Schedule
              </button>
              <button type="button" className="secondary-btn" onClick={onLoadSchedules}>
                Load Schedules
              </button>
            </div>
            {reportSchedules.length > 0 && (
              <ul>
                {reportSchedules.slice(0, 6).map((item) => (
                  <li key={item.id}>
                    <strong>{item.title}</strong> to {item.destination} every {item.interval_minutes} min
                    <div className="button-row">
                      <button type="button" className="secondary-btn" onClick={() => onRunScheduleNow(item.id)}>
                        Run Now
                      </button>
                      <button type="button" className="secondary-btn" onClick={() => onDeleteSchedule(item.id)}>
                        Delete
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
            {reportDeliveries.length > 0 && (
              <div className="aspect-block">
                <h4>Delivery Log</h4>
                <ul>
                  {reportDeliveries.slice(-5).reverse().map((item) => (
                    <li key={item.id}>
                      {item.channel} to {item.destination}: {item.summary}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </article>

          {error && <p className="error">{error}</p>}

          <article className="result stream">
            <h3>Live Sentiment Stream</h3>
            <p className="insight">{loadingLive ? 'Refreshing live updates...' : liveInsight}</p>
            {liveSummary?.negative_spike_alert?.message && <p>{liveSummary.negative_spike_alert.message}</p>}
            <p>
              Tracking Window: <strong>{liveSummary?.window_size || 0}</strong> recent predictions
            </p>
            <p>
              Live Positive: <strong>{liveSummary?.distribution?.positive || 0}</strong>
            </p>
            <p>
              Live Neutral: <strong>{liveSummary?.distribution?.neutral || 0}</strong>
            </p>
            <p>
              Live Negative: <strong>{liveSummary?.distribution?.negative || 0}</strong>
            </p>
            <p>
              Latest Model: <strong>{liveSummary?.latest_event?.model || 'N/A'}</strong>
            </p>
            <label className="live-toggle">
              <input
                type="checkbox"
                checked={liveEnabled}
                onChange={(event) => setLiveEnabled(event.target.checked)}
              />
              Auto refresh every 5 seconds
            </label>
          </article>

          {todayDashboard && (
            <article className="result stream">
              <h3>Today Dashboard</h3>
              <div className="inline-grid">
                {Array.isArray(todayDashboard.cards) && todayDashboard.cards.map((card) => (
                  <div key={card.title} className="aspect-block">
                    <h4>{card.title}</h4>
                    <p><strong>{card.value}</strong></p>
                    <p>{card.subtitle}</p>
                  </div>
                ))}
              </div>
              {Array.isArray(todayDashboard.risk_indicators) && todayDashboard.risk_indicators.length > 0 && (
                <div className="aspect-block">
                  <h4>Risk Indicators</h4>
                  <ul>
                    {todayDashboard.risk_indicators.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
              {Array.isArray(todayDashboard.top_movers) && todayDashboard.top_movers.length > 0 && (
                <div className="aspect-block">
                  <h4>Top Movers</h4>
                  <ul>
                    {todayDashboard.top_movers.map((item) => (
                      <li key={item.topic}>
                        {item.topic}: {item.count} mentions, {item.sentiment_pressure}% pressure
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </article>
          )}
        </section>

        <section className="panel">
          <h2>Results</h2>

          {singleResult && (
            <article className={`result ${singleResult.label}`}>
              <h3>Single Text Output</h3>
              {singleInsight && <p className="insight">{singleInsight}</p>}
              <p>
                Label: <strong>{singleResult.label}</strong>
              </p>
              <p>
                Confidence: <strong>{Math.round(singleResult.confidence * 100)}%</strong>
              </p>
              {singleResult.uncertainty && (
                <p>
                  Uncertainty: <strong>{Math.round(Number(singleResult.uncertainty.score || 0) * 100)}%</strong>{' '}
                  ({singleResult.uncertainty.band})
                </p>
              )}
              {singleResult.language?.label && (
                <p>
                  Language: <strong>{singleResult.language.label}</strong>
                </p>
              )}
              {singleEmotionInsight && <p>{singleEmotionInsight}</p>}
              {sarcasmInsight && <p>{sarcasmInsight}</p>}
              {toxicityInsight && <p>{toxicityInsight}</p>}
              <p>
                Normalized Text: <code>{singleResult.normalized_text}</code>
              </p>
              {Array.isArray(singleResult.aspects) && singleResult.aspects.length > 0 && (
                <div className="aspect-block">
                  <h4>Aspect Breakdown</h4>
                  <p className="aspect-note">
                    This shows sentiment by topic so non-technical users can see what part of the text is positive or negative.
                  </p>
                  <ul>
                    {singleResult.aspects.map((item) => (
                      <li key={item.aspect}>
                        <strong>{item.aspect}:</strong> {item.label} ({Math.round(item.confidence * 100)}%)
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {Array.isArray(singleResult.explainability) && singleResult.explainability.length > 0 && (
                <div className="aspect-block">
                  <h4>Why This Prediction</h4>
                  <div className="topic-grid">
                    {singleResult.explainability.map((item) => (
                      <span key={`${item.word}-${item.direction}`} className="topic-pill">
                        {item.word} ({Math.round(Number(item.impact || 0) * 100)}%)
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </article>
          )}

          {compareData?.results?.length > 0 && (
            <article className="result stream">
              <h3>Model Comparison</h3>
              <p>
                Best Model: <strong>{compareData.best_model}</strong>
              </p>
              <ul>
                {compareData.results.map((item) => (
                  <li key={item.model}>
                    <strong>{item.model}:</strong> {item.label} ({Math.round(Number(item.confidence || 0) * 100)}%)
                  </li>
                ))}
              </ul>
            </article>
          )}

          {csvResult && (
            <article className="result neutral">
              <h3>CSV Batch Output</h3>
              {csvInsight && <p className="insight">{csvInsight.message}</p>}
              <p>
                Predicted Rows: <strong>{csvResult.count}</strong>
              </p>
              <p>
                Positive: <strong>{csvResult.distribution.positive || 0}</strong>
                {csvInsight && <span className="pct"> ({csvInsight.positivePct}%)</span>}
              </p>
              <p>
                Neutral: <strong>{csvResult.distribution.neutral || 0}</strong>
                {csvInsight && <span className="pct"> ({csvInsight.neutralPct}%)</span>}
              </p>
              <p>
                Negative: <strong>{csvResult.distribution.negative || 0}</strong>
                {csvInsight && <span className="pct"> ({csvInsight.negativePct}%)</span>}
              </p>
              {csvEmotionInsight && <p>{csvEmotionInsight}</p>}
              <p>
                Sarcasm Flags: <strong>{csvResult.sarcasm_count || 0}</strong>{' '}
                <span className="pct">({Math.round(Number(csvResult.sarcasm_rate || 0) * 100)}%)</span>
              </p>
              <p>
                Toxic Rows: <strong>{csvResult.toxicity_count || 0}</strong>{' '}
                <span className="pct">(Avg score: {Math.round(Number(csvResult.avg_toxicity_score || 0) * 100)}%)</span>
              </p>
              <p>
                Avg Uncertainty: <strong>{Math.round(Number(csvResult.uncertainty_average || 0) * 100)}%</strong>
              </p>
              {csvResult.data_quality && (
                <p>
                  Data Quality Score: <strong>{Math.round(Number(csvResult.data_quality.quality_score || 0) * 100)}%</strong>{' '}
                  ({csvResult.data_quality.retained_rows}/{csvResult.data_quality.total_rows} rows retained)
                </p>
              )}
              {csvResult.language_distribution && (
                <p>Language Mix: {Object.entries(csvResult.language_distribution).map(([k, v]) => `${k}:${v}`).join(', ')}</p>
              )}
              {Array.isArray(csvResult.competitor_benchmark) && csvResult.competitor_benchmark.length > 0 && (
                <div className="aspect-block">
                  <h4>Competitor Benchmark</h4>
                  <ul>
                    {csvResult.competitor_benchmark.map((item) => (
                      <li key={item.brand}>
                        <strong>{item.brand}:</strong> net {Math.round(Number(item.net_score || 0) * 100)}%
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {csvResult.campaign_impact && (
                <div className="aspect-block">
                  <h4>Campaign Impact</h4>
                  <p>
                    Positive before/after: {Math.round(Number(csvResult.campaign_impact.before_positive_rate || 0) * 100)}% /{' '}
                    {Math.round(Number(csvResult.campaign_impact.after_positive_rate || 0) * 100)}%
                  </p>
                </div>
              )}
              {csvResult.fairness_audit?.groups?.length > 0 && (
                <div className="aspect-block">
                  <h4>Fairness Audit</h4>
                  <p>
                    Positive-rate disparity: {Math.round(Number(csvResult.fairness_audit.positive_rate_disparity || 0) * 100)}%
                  </p>
                </div>
              )}
              <div className="button-row">
                <button type="button" className="secondary-btn" onClick={onGenerateSummary}>
                  Generate Executive Summary
                </button>
                <button type="button" className="secondary-btn" onClick={onSimulateScenario}>
                  Simulate +15% Negative Scenario
                </button>
                <button type="button" className="secondary-btn" onClick={onExportReport}>
                  Export Report Draft
                </button>
              </div>
              {executiveSummary && <p className="insight">{executiveSummary}</p>}
              {scenarioData?.after && (
                <p>
                  Scenario After: positive {Math.round(Number(scenarioData.after.positive || 0) * 100)}%, neutral{' '}
                  {Math.round(Number(scenarioData.after.neutral || 0) * 100)}%, negative {Math.round(Number(scenarioData.after.negative || 0) * 100)}%
                </p>
              )}
              {reportPreview && (
                <pre className="report-preview">{reportPreview}</pre>
              )}
              {reportDownloadUrl && (
                <a className="download-link" href={reportDownloadUrl} download="sentiment-report.pdf">
                  Download PDF Report
                </a>
              )}
            </article>
          )}

          <SentimentChart distribution={chartDistribution} onBarSelect={onSentimentBarSelect} />
          <TrendChart points={trendSeries} />

          {historicalTrends.length > 0 && (
            <article className="result stream">
              <h3>Historical Sentiment Trend</h3>
              <TrendChart
                points={historicalTrends.map((item, index) => ({
                  step: index + 1,
                  positive: Number(item.positive || 0),
                  neutral: Number(item.neutral || 0),
                  negative: Number(item.negative || 0),
                }))}
              />
            </article>
          )}

          {recommendations.length > 0 && (
            <article className="result stream">
              <h3>Actionable Recommendations</h3>
              <ul>
                {recommendations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </article>
          )}

          <article className="result stream">
            <h3>Drill-down Explorer</h3>
            <div className="inline-grid">
              <label>
                Sentiment Label
                <select value={drilldownLabel} onChange={(event) => setDrilldownLabel(event.target.value)}>
                  <option value="">all</option>
                  <option value="positive">positive</option>
                  <option value="neutral">neutral</option>
                  <option value="negative">negative</option>
                </select>
              </label>
              <label>
                Topic Keyword
                <input
                  type="text"
                  value={drilldownTopic}
                  onChange={(event) => setDrilldownTopic(event.target.value)}
                  placeholder="optional keyword"
                />
              </label>
            </div>
            <button type="button" className="secondary-btn" onClick={() => loadDrilldown()} disabled={loadingDrilldown}>
              {loadingDrilldown ? 'Loading...' : 'Load Drill-down'}
            </button>
            {drilldownResults.length > 0 && (
              <ul>
                {drilldownResults.slice(0, 10).map((item, index) => (
                  <li key={`${item.timestamp}-${index}`}>
                    <strong>{item.label}</strong> ({Math.round(Number(item.confidence || 0) * 100)}%) - {item.text || 'No text'}
                  </li>
                ))}
              </ul>
            )}
          </article>

          <GeoHeatmap places={csvResult?.geo_summary || []} sourceColumn={csvResult?.geo_source_column} />
          <TopicClusters topics={csvResult?.topics || []} />

          <article className="result stream">
            <h3>Snapshots And Sharing</h3>
            <label className="field">
              <span>Role</span>
              <select value={snapshotRole} onChange={(event) => setSnapshotRole(event.target.value)}>
                <option value="viewer">viewer</option>
                <option value="analyst">analyst</option>
                <option value="manager">manager</option>
              </select>
            </label>
            <div className="button-row">
              <button type="button" className="secondary-btn" onClick={onSaveSnapshot}>
                Save Snapshot
              </button>
              <button type="button" className="secondary-btn" onClick={onLoadSnapshots}>
                Load Snapshots
              </button>
            </div>
            {snapshots.length > 0 && (
              <ul>
                {snapshots.slice(0, 5).map((item) => (
                  <li key={item.id}>
                    {item.title} ({item.role}) share token: {item.share_token}
                  </li>
                ))}
              </ul>
            )}
          </article>
        </section>
      </main>
    </div>
  )
}
