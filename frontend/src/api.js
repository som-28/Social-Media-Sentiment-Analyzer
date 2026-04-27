const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000'

export async function predictText({ text, model }) {
  const response = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, model }),
  })

  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Prediction request failed')
  }
  return data
}

export async function predictCsv({ file, model }) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('model', model)

  const response = await fetch(`${API_BASE}/api/predict-csv`, {
    method: 'POST',
    body: formData,
  })

  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'CSV prediction request failed')
  }

  return data
}

export async function fetchLiveSummary({ window = 100 } = {}) {
  const response = await fetch(`${API_BASE}/api/live-summary?window=${window}`)
  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.error || 'Live summary request failed')
  }

  return data
}

export async function fetchTrendSeries({ window = 150 } = {}) {
  const response = await fetch(`${API_BASE}/api/trend-series?window=${window}`)
  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.error || 'Trend series request failed')
  }

  return data
}

export async function compareModels({ text, customLexicon }) {
  const response = await fetch(`${API_BASE}/api/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, custom_lexicon: customLexicon }),
  })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Model comparison failed')
  }
  return data
}

export async function simulateScenario({ distribution, deltaNegativePct }) {
  const response = await fetch(`${API_BASE}/api/simulate-scenario`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ distribution, delta_negative_pct: deltaNegativePct }),
  })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Scenario simulation failed')
  }
  return data
}

export async function generateExecutiveSummary({ distribution }) {
  const response = await fetch(`${API_BASE}/api/generate-summary`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ distribution }),
  })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Summary generation failed')
  }
  return data
}

export async function saveSnapshot({ title, role, data }) {
  const response = await fetch(`${API_BASE}/api/save-snapshot`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, role, data }),
  })
  const dataOut = await response.json()
  if (!response.ok) {
    throw new Error(dataOut.error || 'Save snapshot failed')
  }
  return dataOut
}

export async function listSnapshots() {
  const response = await fetch(`${API_BASE}/api/snapshots`)
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Loading snapshots failed')
  }
  return data
}

export async function exportReport({ title, distribution, topics }) {
  const response = await fetch(`${API_BASE}/api/export-report`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, distribution, topics }),
  })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Report export failed')
  }
  return data
}

export async function ingestLiveSource({ sourceType, sourceUrl, model, limit, textField, manualTexts }) {
  const response = await fetch(`${API_BASE}/api/ingest-live`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      source_type: sourceType,
      source_url: sourceUrl,
      model,
      limit,
      text_field: textField,
      manual_texts: manualTexts,
    }),
  })

  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Live source ingestion failed')
  }
  return data
}

export async function listWatchlists() {
  const response = await fetch(`${API_BASE}/api/watchlists`)
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to load watchlists')
  }
  return data
}

export async function createWatchlist({ name, sourceType, sourceUrl, model, limit, textField, manualTexts }) {
  const response = await fetch(`${API_BASE}/api/watchlists`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name,
      source_type: sourceType,
      source_url: sourceUrl,
      model,
      limit,
      text_field: textField,
      manual_texts: manualTexts,
    }),
  })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to create watchlist')
  }
  return data
}

export async function runWatchlist(watchlistId) {
  const response = await fetch(`${API_BASE}/api/watchlists/${watchlistId}/run`, { method: 'POST' })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to run watchlist')
  }
  return data
}

export async function deleteWatchlist(watchlistId) {
  const response = await fetch(`${API_BASE}/api/watchlists/${watchlistId}`, { method: 'DELETE' })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to delete watchlist')
  }
  return data
}

export async function listReportSchedules() {
  const response = await fetch(`${API_BASE}/api/report-schedules`)
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to load schedules')
  }
  return data
}

export async function createReportSchedule({ title, destination, channel, intervalMinutes, distribution, topics }) {
  const response = await fetch(`${API_BASE}/api/report-schedules`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      title,
      destination,
      channel,
      interval_minutes: intervalMinutes,
      distribution,
      topics,
    }),
  })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to create schedule')
  }
  return data
}

export async function runReportScheduleNow(scheduleId) {
  const response = await fetch(`${API_BASE}/api/report-schedules/${scheduleId}/run-now`, { method: 'POST' })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to run schedule')
  }
  return data
}

export async function deleteReportSchedule(scheduleId) {
  const response = await fetch(`${API_BASE}/api/report-schedules/${scheduleId}`, { method: 'DELETE' })
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to delete schedule')
  }
  return data
}

export async function listReportDeliveries() {
  const response = await fetch(`${API_BASE}/api/report-deliveries`)
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to load delivery log')
  }
  return data
}

export async function fetchTodayDashboard({ window = 200 } = {}) {
  const response = await fetch(`${API_BASE}/api/today-dashboard?window=${window}`)
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to load Today dashboard')
  }
  return data
}

export async function fetchHistoricalTrends({ bucket = 'day', window = 1000 } = {}) {
  const response = await fetch(`${API_BASE}/api/historical-trends?bucket=${bucket}&window=${window}`)
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to load historical trends')
  }
  return data
}

export async function fetchDrilldown({ label = '', topic = '', limit = 50 } = {}) {
  const query = new URLSearchParams({
    label,
    topic,
    limit: String(limit),
  })
  const response = await fetch(`${API_BASE}/api/drilldown?${query.toString()}`)
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to load drilldown')
  }
  return data
}

export async function fetchRecommendations({ window = 200 } = {}) {
  const response = await fetch(`${API_BASE}/api/recommendations?window=${window}`)
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Failed to load recommendations')
  }
  return data
}
