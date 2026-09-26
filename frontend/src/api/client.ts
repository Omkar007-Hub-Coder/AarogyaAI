import axios from 'axios'
import type {
  UserProfileCreate,
  UserProfileResponse,
  RecommendationsResponse,
  FullProfileInput,
  PersonalizedResponse,
  FeedbackCreate,
  FeedbackResponse,
  ModelStatus,
  PoseRecognitionResult,
} from '../types'

const api = axios.create({
  baseURL: '/api/v1',
  headers: { 'Content-Type': 'application/json' },
})

// ── Legacy profile endpoints (kept for existing flow) ────────────────────────
export async function createProfile(data: UserProfileCreate): Promise<UserProfileResponse> {
  const res = await api.post<UserProfileResponse>('/profile', data)
  return res.data
}

export async function getRecommendations(userId: number): Promise<RecommendationsResponse> {
  const res = await api.get<RecommendationsResponse>(`/recommendations/${userId}`)
  return res.data
}

// ── Personalization (full ML flow) ────────────────────────────────────────────
export async function getPersonalizedRecommendations(
  profile: FullProfileInput,
  topN = 8,
  mealType?: string,
): Promise<PersonalizedResponse> {
  const payload: Record<string, unknown> = { profile, top_n: topN }
  if (mealType) payload.meal_type = mealType
  const res = await api.post<PersonalizedResponse>('/personalization/recommend', payload)
  return res.data
}

// ── Feedback ──────────────────────────────────────────────────────────────────
export async function submitFeedback(data: FeedbackCreate): Promise<FeedbackResponse> {
  const res = await api.post<FeedbackResponse>('/feedback', data)
  return res.data
}

export async function getUserFeedback(userId: number): Promise<FeedbackResponse[]> {
  const res = await api.get<FeedbackResponse[]>(`/feedback/user/${userId}`)
  return res.data
}

// ── ML model status ───────────────────────────────────────────────────────────
export async function getModelStatus(): Promise<ModelStatus> {
  const res = await api.get<ModelStatus>('/ml/status')
  return res.data
}

// ── Pose recognition (image upload) ──────────────────────────────────────────
export async function recognizePoseFromImage(file: File): Promise<PoseRecognitionResult> {
  const formData = new FormData()
  formData.append('file', file)
  const res = await axios.post<PoseRecognitionResult>('/api/v1/ml/predict/image', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

// ── Health check ─────────────────────────────────────────────────────────────
export async function checkHealth(): Promise<boolean> {
  try {
    await axios.get('/health')
    return true
  } catch {
    return false
  }
}
