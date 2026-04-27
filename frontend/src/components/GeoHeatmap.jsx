const toHeatColor = (score) => {
  if (score >= 0.4) return '#d6f0df'
  if (score >= 0.15) return '#e8f7ed'
  if (score <= -0.4) return '#ffe0de'
  if (score <= -0.15) return '#ffefed'
  return '#fff8e6'
}

export default function GeoHeatmap({ places, sourceColumn }) {
  if (!Array.isArray(places) || places.length === 0) {
    return (
      <div className="chart-card">
        <h3>Geo Sentiment Heatmap</h3>
        <p className="muted-copy">
          Add a location column (for example city, country, region, or location) in CSV to see geography insights.
        </p>
      </div>
    )
  }

  return (
    <div className="chart-card">
      <h3>Geo Sentiment Heatmap</h3>
      <p className="muted-copy">Source column: {sourceColumn || 'location'}</p>
      <div className="geo-grid">
        {places.map((item) => (
          <article
            className="geo-cell"
            key={item.place}
            style={{ background: toHeatColor(Number(item.net_score || 0)) }}
          >
            <h4>{item.place}</h4>
            <p>Total: {item.count}</p>
            <p>Positive: {item.positive}</p>
            <p>Neutral: {item.neutral}</p>
            <p>Negative: {item.negative}</p>
            <p>Net score: {item.net_score}</p>
          </article>
        ))}
      </div>
    </div>
  )
}
