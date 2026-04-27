import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const COLORS = {
  positive: '#1f9d65',
  neutral: '#b08900',
  negative: '#c33232',
}

export default function SentimentChart({ distribution, onBarSelect }) {
  const data = ['positive', 'neutral', 'negative'].map((label) => ({
    name: label,
    value: distribution[label] || 0,
  }))

  return (
    <div className="chart-card">
      <h3>Sentiment Distribution</h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 16, right: 20, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis dataKey="name" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Bar
            dataKey="value"
            radius={[8, 8, 0, 0]}
            cursor={onBarSelect ? 'pointer' : 'default'}
            onClick={(payload) => {
              if (!onBarSelect || !payload || !payload.name) {
                return
              }
              onBarSelect(payload.name)
            }}
          >
            {data.map((entry) => (
              <Cell key={entry.name} fill={COLORS[entry.name]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
