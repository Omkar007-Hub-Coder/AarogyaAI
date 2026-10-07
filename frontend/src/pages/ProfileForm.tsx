import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getPersonalizedRecommendations } from '../api/client'
import type { FullProfileInput, PrimaryGoal, YogaExperience, DietaryPreference, PersonalizedResponse } from '../types'

const INITIAL: FullProfileInput = {
  name: '',
  age: 28,
  gender: 'other',
  height_cm: 165,
  weight_kg: 65,
  activity_level: 'moderate',
  primary_goal: 'General Fitness',
  yoga_experience: 'none',
  fitness_experience: 'none',
  flexibility_score: 3,
  strength_score: 3,
  balance_score: 3,
  session_duration_min: 30,
  days_per_week: 3,
  dietary_preference: 'none',
  food_preferences: [],
  allergies: [],
  sleep_hours: 7,
  previous_performance_score: 5,
}

function computeBMI(h: number, w: number): string {
  if (h <= 0 || w <= 0) return '—'
  const bmi = w / ((h / 100) ** 2)
  return bmi.toFixed(1)
}

function bmiCategory(bmiStr: string): string {
  const b = parseFloat(bmiStr)
  if (isNaN(b)) return ''
  if (b < 18.5) return 'Underweight'
  if (b < 25) return 'Normal weight'
  if (b < 30) return 'Overweight'
  return 'Obese'
}

interface SliderFieldProps {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  leftLabel?: string
  rightLabel?: string
}

function SliderField({ label, value, min, max, step = 1, onChange, leftLabel, rightLabel }: SliderFieldProps) {
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <label className="text-sm font-medium text-gray-700">{label}</label>
        <span className="text-sm font-semibold text-green-700">{value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-600"
      />
      {(leftLabel || rightLabel) && (
        <div className="flex justify-between text-xs text-gray-400 mt-0.5">
          <span>{leftLabel}</span><span>{rightLabel}</span>
        </div>
      )}
    </div>
  )
}

export default function ProfileForm() {
  const navigate = useNavigate()
  const [form, setForm] = useState<FullProfileInput>(INITIAL)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [allergyInput, setAllergyInput] = useState('')

  const bmi = computeBMI(form.height_cm, form.weight_kg)
  const cat = bmiCategory(bmi)

  function set<K extends keyof FullProfileInput>(key: K, value: FullProfileInput[K]) {
    setForm(prev => ({ ...prev, [key]: value }))
  }

  function addAllergy() {
    const val = allergyInput.trim().toLowerCase()
    if (val && !form.allergies.includes(val)) {
      set('allergies', [...form.allergies, val])
    }
    setAllergyInput('')
  }

  function removeAllergy(a: string) {
    set('allergies', form.allergies.filter(x => x !== a))
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const result: PersonalizedResponse = await getPersonalizedRecommendations(form, 8)
      // Store result in sessionStorage so Dashboard can read it
      sessionStorage.setItem('aarogya_result', JSON.stringify(result))
      sessionStorage.setItem('aarogya_profile', JSON.stringify(form))
      navigate('/dashboard')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      setError(`Could not get recommendations. Make sure the backend is running on port 8000. (${msg})`)
    } finally {
      setLoading(false)
    }
  }

  const selectCls = "w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 bg-white"
  const inputCls  = "w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500"

  return (
    <div className="max-w-2xl mx-auto pb-12">
      <h2 className="text-2xl font-bold text-gray-900 mb-1">Your Health Profile</h2>
      <p className="text-sm text-gray-500 mb-6">All fields help personalise your recommendations. Only Name, Age, Height, Weight, Goal, and Activity Level are required.</p>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-4 mb-6 text-sm">{error}</div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">

        {/* Section 1: Basic */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm space-y-4">
          <h3 className="font-semibold text-gray-800 text-base border-b pb-2">Basic Information</h3>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Name *</label>
            <input type="text" required value={form.name}
              onChange={e => set('name', e.target.value)} className={inputCls} placeholder="Your name" />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Age *</label>
              <input type="number" min={5} max={120} required value={form.age}
                onChange={e => set('age', Number(e.target.value))} className={inputCls} />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Gender</label>
              <select value={form.gender} onChange={e => set('gender', e.target.value as FullProfileInput['gender'])} className={selectCls}>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Prefer not to say</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Height (cm) *</label>
              <input type="number" min={50} max={300} step={0.5} required value={form.height_cm}
                onChange={e => set('height_cm', Number(e.target.value))} className={inputCls} />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Weight (kg) *</label>
              <input type="number" min={10} max={500} step={0.5} required value={form.weight_kg}
                onChange={e => set('weight_kg', Number(e.target.value))} className={inputCls} />
            </div>
          </div>

          {/* Live BMI */}
          {bmi !== '—' && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-sm flex items-center gap-3">
              <span className="font-semibold text-green-700">BMI: {bmi}</span>
              <span className="text-green-600">({cat})</span>
              <span className="text-gray-400 text-xs ml-auto">WHO classification — not a medical diagnosis</span>
            </div>
          )}
        </div>

        {/* Section 2: Goals & Activity */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm space-y-4">
          <h3 className="font-semibold text-gray-800 text-base border-b pb-2">Goals &amp; Activity</h3>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Primary Goal *</label>
            <select value={form.primary_goal} onChange={e => set('primary_goal', e.target.value as PrimaryGoal)} className={selectCls}>
              <option value="General Fitness">General Fitness</option>
              <option value="Flexibility">Flexibility</option>
              <option value="Strength">Strength</option>
              <option value="Mobility">Mobility</option>
              <option value="Weight Management">Weight Management</option>
              <option value="Stress Reduction">Stress Reduction</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Activity Level *</label>
            <select value={form.activity_level} onChange={e => set('activity_level', e.target.value as FullProfileInput['activity_level'])} className={selectCls}>
              <option value="sedentary">Sedentary (desk job, little exercise)</option>
              <option value="light">Light (1–3 days/week)</option>
              <option value="moderate">Moderate (3–5 days/week)</option>
              <option value="active">Active (6–7 days/week)</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Yoga Experience</label>
              <select value={form.yoga_experience} onChange={e => set('yoga_experience', e.target.value as YogaExperience)} className={selectCls}>
                <option value="none">None</option>
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Fitness Experience</label>
              <select value={form.fitness_experience} onChange={e => set('fitness_experience', e.target.value as YogaExperience)} className={selectCls}>
                <option value="none">None</option>
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Session Duration (min)</label>
              <select value={form.session_duration_min} onChange={e => set('session_duration_min', Number(e.target.value))} className={selectCls}>
                {[15, 20, 30, 45, 60, 90].map(v => <option key={v} value={v}>{v} min</option>)}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Days per week</label>
              <select value={form.days_per_week} onChange={e => set('days_per_week', Number(e.target.value))} className={selectCls}>
                {[1,2,3,4,5,6,7].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
          </div>
        </div>

        {/* Section 3: Fitness Scores */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm space-y-4">
          <h3 className="font-semibold text-gray-800 text-base border-b pb-2">Self-Assessment</h3>
          <SliderField label="Flexibility" value={form.flexibility_score} min={1} max={5} onChange={v => set('flexibility_score', v)} leftLabel="Very stiff" rightLabel="Very flexible" />
          <SliderField label="Strength" value={form.strength_score} min={1} max={5} onChange={v => set('strength_score', v)} leftLabel="Low" rightLabel="High" />
          <SliderField label="Balance" value={form.balance_score} min={1} max={5} onChange={v => set('balance_score', v)} leftLabel="Poor" rightLabel="Excellent" />
          <SliderField label="Sleep (hours/night)" value={form.sleep_hours} min={1} max={12} step={0.5} onChange={v => set('sleep_hours', v)} leftLabel="Very little" rightLabel="Plenty" />
          <SliderField label="Recent Performance (self-rated)" value={form.previous_performance_score} min={0} max={10} onChange={v => set('previous_performance_score', v)} leftLabel="Poor" rightLabel="Excellent" />
        </div>

        {/* Section 4: Diet & Allergies */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm space-y-4">
          <h3 className="font-semibold text-gray-800 text-base border-b pb-2">Diet &amp; Allergies</h3>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Dietary Preference</label>
            <select value={form.dietary_preference} onChange={e => set('dietary_preference', e.target.value as DietaryPreference)} className={selectCls}>
              <option value="none">No preference</option>
              <option value="vegetarian">Vegetarian</option>
              <option value="vegan">Vegan</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Allergies / Intolerances <span className="text-gray-400 font-normal">(hard filter — these items will be excluded)</span>
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={allergyInput}
                onChange={e => setAllergyInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); addAllergy() } }}
                placeholder="e.g. gluten, dairy, nuts"
                className={inputCls + ' flex-1'}
              />
              <button type="button" onClick={addAllergy}
                className="px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md text-sm border border-gray-300">
                Add
              </button>
            </div>
            {form.allergies.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-2">
                {form.allergies.map(a => (
                  <span key={a} className="inline-flex items-center gap-1 bg-red-50 border border-red-200 text-red-700 text-xs px-2 py-1 rounded-full">
                    {a}
                    <button type="button" onClick={() => removeAllergy(a)} className="hover:text-red-900 ml-0.5">×</button>
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-green-600 hover:bg-green-700 disabled:bg-green-300 text-white font-semibold py-3 rounded-lg transition-colors"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
              </svg>
              Computing recommendations…
            </span>
          ) : 'Get Personalised Recommendations →'}
        </button>
      </form>
    </div>
  )
}
