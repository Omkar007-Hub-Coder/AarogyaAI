import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getRecommendations } from '../api/client'
import type { RecommendationsResponse, RecommendationItem } from '../types'

interface SectionProps {
  title: string
  emoji: string
  items: RecommendationItem[]
}

function Section({ title, emoji, items }: SectionProps) {
  if (items.length === 0) {
    return (
      <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
        <h3 className="font-semibold text-gray-800 mb-2">{emoji} {title}</h3>
        <p className="text-sm text-gray-400">No recommendations found for your profile in this category.</p>
      </div>
    )
  }
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
      <h3 className="font-semibold text-gray-800 mb-4">{emoji} {title}</h3>
      <ul className="space-y-3">
        {items.map(item => (
          <li key={item.id} className="flex items-start gap-3">
            <span className="mt-0.5 h-2 w-2 rounded-full bg-green-500 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-gray-800">{item.item_name}</p>
              {item.description && (
                <p className="text-xs text-gray-500">{item.description}</p>
              )}
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default function Recommendations() {
  const { userId } = useParams<{ userId: string }>()
  const [data, setData] = useState<RecommendationsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!userId) return
    getRecommendations(Number(userId))
      .then(setData)
      .catch(() => setError('Failed to load recommendations. Is the backend running?'))
      .finally(() => setLoading(false))
  }, [userId])

  if (loading) {
    return <p className="text-center text-gray-500 py-12">Loading recommendations…</p>
  }

  if (error || !data) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600 mb-4">{error ?? 'Something went wrong.'}</p>
        <Link to="/profile" className="text-green-600 hover:underline text-sm">← Back to profile</Link>
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Your Recommendations</h2>
        <Link to="/profile" className="text-sm text-green-600 hover:underline">← New profile</Link>
      </div>

      <div className="space-y-5">
        <Section title="Yoga" emoji="🧘" items={data.yoga} />
        <Section title="Fitness" emoji="💪" items={data.fitness} />
        <Section title="Ahar (Diet)" emoji="🥗" items={data.ahar} />
      </div>
    </div>
  )
}
