const NOISY_TOPICS = new Set(['amp', 'http', 'https', 'www', 'com', 'rt', 'tco', 'co'])

function formatCount(value) {
  return new Intl.NumberFormat('en-US').format(Number(value || 0))
}

export default function TopicClusters({ topics }) {
  const cleanedTopics = Array.isArray(topics)
    ? topics
        .map((item) => ({
          topic: String(item.topic || '').trim(),
          count: Number(item.count || 0),
          keywords: Array.isArray(item.keywords) ? item.keywords : [],
          positive: Number(item.positive || 0),
          neutral: Number(item.neutral || 0),
          negative: Number(item.negative || 0),
        }))
        .filter((item) => item.topic && item.count > 0 && !NOISY_TOPICS.has(item.topic.toLowerCase()))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10)
    : []

  if (cleanedTopics.length === 0) {
    return (
      <div className="chart-card">
        <h3>Topic Clusters</h3>
        <p className="muted-copy">Upload CSV data to generate frequent topic keywords from your posts.</p>
      </div>
    )
  }

  const maxCount = Math.max(...cleanedTopics.map((item) => item.count), 1)

  return (
    <div className="chart-card">
      <h3>Topic Clusters</h3>
      <p className="muted-copy">Most discussed themes from your processed posts.</p>
      <div className="topic-grid">
        {cleanedTopics.map((item) => (
          <span
            className="topic-pill"
            key={item.topic}
            style={{ '--topic-strength': Math.max(0.2, item.count / maxCount) }}
            title={`${item.topic}: ${formatCount(item.count)} mentions`}
          >
            <span className="topic-label">{item.topic}</span>
            <span className="topic-count">{formatCount(item.count)}</span>
            {item.keywords.length > 0 && (
              <span className="topic-meta">{item.keywords.join(' • ')}</span>
            )}
          </span>
        ))}
      </div>
    </div>
  )
}
