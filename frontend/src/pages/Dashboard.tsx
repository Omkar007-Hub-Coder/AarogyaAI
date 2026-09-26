import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import type { PersonalizedResponse, YogaRecItem, AharRecItem, FullProfileInput } from '../types'
import FeedbackModal from '../components/FeedbackModal'

function DifficultyBadge({ d }: { d: string }) {
  const color =
    d === 'Beginner' ? 'bg-green-100 text-green-700 border-green-200' :
    d === 'Intermediate' ? 'bg-yellow-100 text-yellow-700 border-yellow-200' :
    'bg-red-100 text-red-700 border-red-200'
  return <span className={`text-xs border px-2 py-0.5 rounded-full font-medium ${color}`}>{d}</span>
}

function ScoreBar({ score, label }: { score: number; label: string }) {
  const pct = Math.round(score * 100)
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-gray-500 w-16 shrink-0">{label}</span>
      <div className="flex-1 bg-gray-100 rounded-full h-1.5">
        <div className="bg-green-500 h-1.5 rounded-full" style={{ width: `${pct}%` }} />
      </div>
      <span className="text-gray-600 font-medium w-8 text-right">{pct}%</span>
    </div>
  )
}

function MacroBar({ label, g, pct }: { label: string; g: number; pct: number }) {
  return (
    <div>
      <div className="flex justify-between text-xs text-gray-600 mb-0.5">
        <span>{label}</span><span className="font-medium">{g}g <span className="text-gray-400">({Math.round(pct*100)}%)</span></span>
      </div>
      <div className="bg-gray-100 rounded-full h-2">
        <div className="bg-green-500 h-2 rounded-full" style={{ width: `${Math.round(pct*100)}%` }} />
      </div>
    </div>
  )
}

export default function Dashboard() {
  const navigate = useNavigate()
  const [data, setData] = useState<PersonalizedResponse | null>(null)
  const [, setProfile] = useState<FullProfileInput | null>(null)
  const [feedbackTarget, setFeedbackTarget] = useState<{ category: 'yoga' | 'ahar'; name: string } | null>(null)
  const [feedbackDone, setFeedbackDone] = useState<Set<string>>(new Set())
  const [activeTab, setActiveTab] = useState<'yoga' | 'ahar' | 'summary'>('summary')

  useEffect(() => {
    const raw = sessionStorage.getItem('aarogya_result')
    const prof = sessionStorage.getItem('aarogya_profile')
    if (!raw) { navigate('/profile'); return }
    setData(JSON.parse(raw))
    if (prof) setProfile(JSON.parse(prof))
  }, [navigate])

  if (!data) return <p className="text-center text-gray-500 py-12">Loading…</p>

  const { profile_summary, bmi_info, macro_estimate, goal_prediction, difficulty_prediction, yoga_recommendations, ahar_recommendations, data_integrity } = data
  const m = macro_estimate.macro_targets

  return (
    <div className="max-w-2xl mx-auto pb-12">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Your Recommendations</h2>
          <p className="text-sm text-gray-500">Personalised for {profile_summary.name}</p>
        </div>
        <Link to="/profile" className="text-sm text-green-600 hover:underline">← New profile</Link>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 mb-6 gap-1">
        {(['summary','yoga','ahar'] as const).map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors capitalize ${
              activeTab === tab ? 'border-green-600 text-green-700' : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}>
            {tab === 'summary' ? 'Profile Summary' : tab === 'yoga' ? `Yoga (${yoga_recommendations.length})` : `Ahar (${ahar_recommendations.length})`}
          </button>
        ))}
      </div>

      {/* Summary Tab */}
      {activeTab === 'summary' && (
        <div className="space-y-5">
          {/* BMI card */}
          <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
            <h3 className="font-semibold text-gray-800 mb-3">Body Metrics</h3>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div className="bg-green-50 rounded-lg p-3">
                <div className="text-2xl font-bold text-green-700">{bmi_info.bmi.toFixed(1)}</div>
                <div className="text-xs text-gray-500 mt-0.5">BMI</div>
                <div className="text-xs font-medium text-green-600 mt-1">{bmi_info.category}</div>
              </div>
              <div className="bg-blue-50 rounded-lg p-3">
                <div className="text-2xl font-bold text-blue-700">{macro_estimate.tdee_kcal.toFixed(0)}</div>
                <div className="text-xs text-gray-500 mt-0.5">TDEE (kcal/day)</div>
                <div className="text-xs text-blue-600 mt-1">Estimate</div>
              </div>
              <div className="bg-purple-50 rounded-lg p-3">
                <div className="text-xl font-bold text-purple-700">{profile_summary.primary_goal}</div>
                <div className="text-xs text-gray-500 mt-0.5">Goal</div>
                <div className="text-xs text-purple-600 mt-1">{profile_summary.activity_level}</div>
              </div>
            </div>
            <p className="text-xs text-gray-400 mt-3 text-center">{bmi_info.estimate_note}</p>
          </div>

          {/* Macro targets */}
          <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
            <div className="flex items-start justify-between mb-3">
              <h3 className="font-semibold text-gray-800">Estimated Daily Macro Targets</h3>
              <span className="text-xs bg-amber-100 text-amber-700 border border-amber-200 px-2 py-0.5 rounded-full">Estimate</span>
            </div>
            <div className="space-y-3">
              <MacroBar label="Protein" g={m.protein_g} pct={m.protein_pct} />
              <MacroBar label="Carbohydrates" g={m.carbs_g} pct={m.carbs_pct} />
              <MacroBar label="Fat" g={m.fat_g} pct={m.fat_pct} />
            </div>
            <p className="text-xs text-gray-400 mt-3">{macro_estimate.estimate_label}</p>
          </div>

          {/* ML predictions */}
          {(goal_prediction || difficulty_prediction) && (
            <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
              <div className="flex items-center gap-2 mb-3">
                <h3 className="font-semibold text-gray-800">ML Predictions</h3>
                <span className="text-xs bg-orange-100 text-orange-700 border border-orange-200 px-2 py-0.5 rounded-full">⚠ Synthetic prototype</span>
              </div>
              <div className="space-y-2 text-sm">
                {goal_prediction?.predicted_goal && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Suggested goal</span>
                    <span className="font-medium">{goal_prediction.predicted_goal}
                      {goal_prediction.confidence !== undefined && <span className="text-gray-400 ml-1 text-xs">({(goal_prediction.confidence*100).toFixed(0)}% conf)</span>}
                    </span>
                  </div>
                )}
                {difficulty_prediction?.predicted_difficulty && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Suggested difficulty</span>
                    <DifficultyBadge d={difficulty_prediction.predicted_difficulty} />
                  </div>
                )}
              </div>
              <p className="text-xs text-gray-400 mt-2">These predictions are from models trained on synthetic data. They are informational only and not clinically validated.</p>
            </div>
          )}

          {/* Profile detail */}
          <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
            <h3 className="font-semibold text-gray-800 mb-3">Profile Details</h3>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
              {([
                ['Yoga experience', profile_summary.yoga_experience],
                ['Activity level', profile_summary.activity_level],
                ['Dietary preference', profile_summary.dietary_preference],
                ['Session duration', `${profile_summary.session_duration_min} min`],
              ] as [string, string][]).map(([k, v]) => (
                <div key={k}><dt className="text-gray-500 text-xs">{k}</dt><dd className="font-medium text-gray-800 capitalize">{v}</dd></div>
              ))}
              {profile_summary.allergies.length > 0 && (
                <div className="col-span-2"><dt className="text-gray-500 text-xs">Allergies (hard-filtered)</dt>
                  <dd className="flex gap-1 mt-0.5 flex-wrap">{profile_summary.allergies.map(a => <span key={a} className="bg-red-50 text-red-700 text-xs px-2 py-0.5 rounded-full border border-red-200">{a}</span>)}</dd>
                </div>
              )}
            </dl>
          </div>

          {/* Navigation shortcuts */}
          <div className="grid grid-cols-2 gap-4">
            <button onClick={() => setActiveTab('yoga')} className="bg-green-600 hover:bg-green-700 text-white font-medium py-3 rounded-lg text-sm">View Yoga Poses →</button>
            <button onClick={() => setActiveTab('ahar')} className="bg-emerald-600 hover:bg-emerald-700 text-white font-medium py-3 rounded-lg text-sm">View Ahar Plan →</button>
          </div>
        </div>
      )}

      {/* Yoga Tab */}
      {activeTab === 'yoga' && (
        <div className="space-y-4">
          <p className="text-xs text-gray-500 bg-green-50 border border-green-200 rounded-lg p-3">
            Ranked by feature-vector suitability score. Method: content-based ranking. No user-outcome data used.
          </p>
          {yoga_recommendations.length === 0 && (
            <div className="bg-white border border-gray-200 rounded-xl p-6 text-center text-gray-400">No yoga recommendations found for this profile.</div>
          )}
          {yoga_recommendations.map((pose: YogaRecItem, i: number) => (
            <div key={pose.pose_name} className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-400 font-mono w-5">#{i+1}</span>
                  <h4 className="font-semibold text-gray-900 text-sm">{pose.pose_name}</h4>
                </div>
                <DifficultyBadge d={pose.difficulty} />
              </div>
              <ScoreBar score={pose.suitability_score} label="Suitability" />
              <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-500">
                <span>⏱ {pose.duration_min} min</span>
                <span>📂 {pose.category}</span>
                <span>🎯 {pose.target_areas.slice(0,3).join(', ')}</span>
              </div>
              <p className="text-xs text-gray-600 mt-2 italic">{pose.reason}</p>
              <div className="flex items-center justify-between mt-3">
                {feedbackDone.has(pose.pose_name)
                  ? <span className="text-xs text-green-600">✓ Feedback submitted</span>
                  : <button onClick={() => setFeedbackTarget({ category: 'yoga', name: pose.pose_name })}
                      className="text-xs text-gray-400 hover:text-green-600 border border-gray-200 hover:border-green-300 px-3 py-1 rounded-full transition-colors">
                      Rate this pose
                    </button>
                }
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Ahar Tab */}
      {activeTab === 'ahar' && (
        <div className="space-y-4">
          <p className="text-xs text-gray-500 bg-emerald-50 border border-emerald-200 rounded-lg p-3">
            Content-based ranking using curated food metadata (ICMR-NIN / USDA reference values). Not a trained ML model. Allergies are hard-filtered.
          </p>
          {ahar_recommendations.length === 0 && (
            <div className="bg-white border border-gray-200 rounded-xl p-6 text-center text-gray-400">No food recommendations match your dietary preferences and allergy filters.</div>
          )}
          {ahar_recommendations.map((food: AharRecItem, i: number) => (
            <div key={food.food_name} className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm">
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-400 font-mono w-5">#{i+1}</span>
                  <h4 className="font-semibold text-gray-900 text-sm">{food.food_name}</h4>
                </div>
                <span className="text-xs bg-gray-100 text-gray-600 border border-gray-200 px-2 py-0.5 rounded-full capitalize">{food.dietary_category}</span>
              </div>
              <ScoreBar score={food.relevance_score} label="Relevance" />
              <div className="mt-3 grid grid-cols-4 gap-2 text-center text-xs">
                {[
                  { label: 'Calories', val: food.calories_kcal, unit: 'kcal' },
                  { label: 'Protein', val: food.protein_g, unit: 'g' },
                  { label: 'Carbs', val: food.carbs_g, unit: 'g' },
                  { label: 'Fat', val: food.fat_g, unit: 'g' },
                ].map(({ label, val, unit }) => (
                  <div key={label} className="bg-gray-50 rounded-lg py-2 border border-gray-100">
                    <div className="font-semibold text-gray-800">{val}{unit}</div>
                    <div className="text-gray-400 text-[10px]">{label}</div>
                  </div>
                ))}
              </div>
              <div className="flex flex-wrap gap-1 mt-2">
                {food.meal_suitability.map(m => <span key={m} className="text-[10px] bg-blue-50 text-blue-600 border border-blue-100 px-1.5 py-0.5 rounded-full capitalize">{m}</span>)}
              </div>
              <p className="text-xs text-gray-600 mt-2 italic">{food.reason}</p>
              <p className="text-xs text-gray-400 mt-1">Serving: {food.serving_size_g}g</p>
              <div className="mt-3">
                {feedbackDone.has(food.food_name)
                  ? <span className="text-xs text-green-600">✓ Feedback submitted</span>
                  : <button onClick={() => setFeedbackTarget({ category: 'ahar', name: food.food_name })}
                      className="text-xs text-gray-400 hover:text-green-600 border border-gray-200 hover:border-green-300 px-3 py-1 rounded-full transition-colors">
                      Rate this food
                    </button>
                }
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Feedback Modal */}
      {feedbackTarget && (
        <FeedbackModal
          category={feedbackTarget.category}
          itemName={feedbackTarget.name}
          onClose={() => setFeedbackTarget(null)}
          onSubmit={() => {
            setFeedbackDone(prev => new Set([...prev, feedbackTarget.name]))
            setFeedbackTarget(null)
          }}
        />
      )}

      {/* Data integrity footer */}
      <details className="mt-8 text-xs text-gray-400">
        <summary className="cursor-pointer hover:text-gray-600">Data provenance &amp; methodology</summary>
        <div className="mt-2 space-y-1 bg-gray-50 border border-gray-200 rounded-lg p-3">
          {Object.entries(data_integrity).map(([k, v]) => (
            <div key={k}><span className="font-medium text-gray-500 capitalize">{k.replace(/_/g,' ')}:</span> {v}</div>
          ))}
        </div>
      </details>
    </div>
  )
}
