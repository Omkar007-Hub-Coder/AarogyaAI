import { useState } from 'react'
import { submitFeedback } from '../api/client'
import type { FeedbackCreate } from '../types'

interface Props {
  category: 'yoga' | 'fitness' | 'ahar'
  itemName: string
  userId?: number
  recommendationId?: number
  onClose: () => void
  onSubmit: () => void
}

export default function FeedbackModal({ category, itemName, userId, recommendationId, onClose, onSubmit }: Props) {
  const [completed, setCompleted] = useState<0 | 1 | undefined>(undefined)
  const [difficulty, setDifficulty] = useState<number>(3)
  const [comfort, setComfort] = useState<number>(3)
  const [satisfaction, setSatisfaction] = useState<number>(3)
  const [notes, setNotes] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      // If no userId, submit as anonymous feedback stored locally only
      if (!userId) {
        // Just call onSubmit to mark as done (no backend call without userId)
        onSubmit()
        return
      }
      const payload: FeedbackCreate = {
        user_id: userId,
        category,
        item_name: itemName,
        recommendation_id: recommendationId,
        completed,
        perceived_difficulty: difficulty,
        comfort,
        satisfaction,
        notes: notes.trim() || undefined,
      }
      await submitFeedback(payload)
      onSubmit()
    } catch (err) {
      setError('Could not submit feedback. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  function StarRating({ value, onChange, label }: { value: number; onChange: (v: number) => void; label: string }) {
    return (
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
        <div className="flex gap-1">
          {[1,2,3,4,5].map(n => (
            <button key={n} type="button" onClick={() => onChange(n)}
              className={`text-xl transition-colors ${n <= value ? 'text-yellow-400' : 'text-gray-200 hover:text-yellow-200'}`}>
              ★
            </button>
          ))}
          <span className="text-xs text-gray-400 ml-2 self-center">{value}/5</span>
        </div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div className="bg-white rounded-xl shadow-lg max-w-sm w-full p-6" onClick={e => e.stopPropagation()}>
        <div className="flex justify-between items-start mb-4">
          <div>
            <h3 className="font-semibold text-gray-900">Rate: {itemName}</h3>
            <p className="text-xs text-gray-500 mt-0.5 capitalize">{category}</p>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-xl leading-none">×</button>
        </div>

        {error && <div className="bg-red-50 text-red-600 text-xs p-2 rounded-lg mb-3">{error}</div>}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Did you complete this?</label>
            <div className="flex gap-3">
              {([1, 0] as const).map(v => (
                <button key={v} type="button" onClick={() => setCompleted(v)}
                  className={`flex-1 py-2 text-sm rounded-lg border transition-colors ${completed === v ? (v === 1 ? 'bg-green-100 border-green-400 text-green-700 font-semibold' : 'bg-red-100 border-red-400 text-red-700 font-semibold') : 'border-gray-200 text-gray-600 hover:bg-gray-50'}`}>
                  {v === 1 ? '✓ Yes' : '✗ No'}
                </button>
              ))}
            </div>
          </div>

          <StarRating value={satisfaction} onChange={setSatisfaction} label="Overall satisfaction" />
          <StarRating value={difficulty} onChange={setDifficulty} label="Perceived difficulty (1=easy, 5=hard)" />
          <StarRating value={comfort} onChange={setComfort} label="Comfort level" />

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Notes (optional)</label>
            <textarea value={notes} onChange={e => setNotes(e.target.value)} rows={2}
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 resize-none"
              placeholder="Any comments…" />
          </div>

          <p className="text-xs text-gray-400">
            Feedback applies a transparent heuristic adjustment to future ranking. This is not reinforcement learning.
          </p>

          <button type="submit" disabled={loading}
            className="w-full bg-green-600 hover:bg-green-700 disabled:bg-green-300 text-white font-medium py-2 rounded-lg text-sm transition-colors">
            {loading ? 'Submitting…' : 'Submit Feedback'}
          </button>
        </form>
      </div>
    </div>
  )
}
