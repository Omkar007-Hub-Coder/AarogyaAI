// All TypeScript types for AarogyaAI frontend

// ── Legacy profile (used by existing /api/v1/profile endpoint) ──────────────
export interface UserProfileCreate {
  name: string
  age: number
  gender: 'male' | 'female' | 'other'
  height_cm: number
  weight_kg: number
  activity_level: 'sedentary' | 'light' | 'moderate' | 'active'
  health_goal: 'weight_loss' | 'muscle_gain' | 'flexibility' | 'general_wellness'
  dietary_preference: 'vegetarian' | 'vegan' | 'non-vegetarian' | 'none'
}

export interface UserProfileResponse extends UserProfileCreate {
  id: number
  created_at: string
}

export interface RecommendationItem {
  id: number
  category: string
  item_name: string
  description: string | null
  score: number
  feedback_adjusted_score: number | null
}

export interface RecommendationsResponse {
  user_id: number
  yoga: RecommendationItem[]
  fitness: RecommendationItem[]
  ahar: RecommendationItem[]
}

// ── Full profile (personalization endpoint) ──────────────────────────────────
export type PrimaryGoal =
  | 'Flexibility'
  | 'Strength'
  | 'Mobility'
  | 'Weight Management'
  | 'Stress Reduction'
  | 'General Fitness'

export type YogaExperience = 'none' | 'beginner' | 'intermediate' | 'advanced'
export type DietaryPreference = 'vegetarian' | 'vegan' | 'non-vegetarian' | 'none'

export interface FullProfileInput {
  name: string
  age: number
  gender: 'male' | 'female' | 'other'
  height_cm: number
  weight_kg: number
  activity_level: 'sedentary' | 'light' | 'moderate' | 'active'
  primary_goal: PrimaryGoal
  yoga_experience: YogaExperience
  fitness_experience: YogaExperience
  flexibility_score: number    // 1–5
  strength_score: number       // 1–5
  balance_score: number        // 1–5
  session_duration_min: number
  days_per_week: number
  dietary_preference: DietaryPreference
  food_preferences: string[]
  allergies: string[]
  sleep_hours: number
  previous_performance_score: number  // 0–10
}

export interface BMIInfo {
  bmi: number
  category: string
  estimate_note: string
}

export interface MacroTargets {
  protein_g: number
  carbs_g: number
  fat_g: number
  protein_pct: number
  carbs_pct: number
  fat_pct: number
}

export interface MacroEstimate {
  estimate_label: string
  bmr_kcal: number
  tdee_kcal: number
  activity_level: string
  pal_used: number
  goal: string
  macro_targets: MacroTargets
}

export interface YogaRecItem {
  pose_name: string
  suitability_score: number
  difficulty: string
  duration_min: number
  target_areas: string[]
  category: string
  reason: string
}

export interface AharRecItem {
  food_name: string
  relevance_score: number
  dietary_category: string
  serving_size_g: number
  calories_kcal: number
  protein_g: number
  carbs_g: number
  fat_g: number
  fibre_g: number
  meal_suitability: string[]
  goal_alignment: string[]
  reason: string
}

export interface PersonalizedResponse {
  data_integrity: Record<string, string>
  profile_summary: {
    name: string
    age: number
    gender: string
    bmi: number
    bmi_category: string
    primary_goal: string
    activity_level: string
    yoga_experience: string
    dietary_preference: string
    allergies: string[]
    session_duration_min: number
  }
  bmi_info: BMIInfo
  macro_estimate: MacroEstimate
  goal_prediction: { predicted_goal?: string; confidence?: number; data_source: string; note?: string } | null
  difficulty_prediction: { predicted_difficulty?: string; data_source: string; note?: string } | null
  yoga_recommendations: YogaRecItem[]
  ahar_recommendations: AharRecItem[]
}

// ── Pose recognition ─────────────────────────────────────────────────────────
export interface PoseRecognitionResult {
  predicted_pose: string
  confidence: number
  top_k: Array<{ pose: string; confidence: number }>
}

// ── Feedback ─────────────────────────────────────────────────────────────────
export interface FeedbackCreate {
  user_id: number
  category: 'yoga' | 'fitness' | 'ahar'
  item_name: string
  recommendation_id?: number
  completed?: 0 | 1
  perceived_difficulty?: number   // 1–5
  comfort?: number                 // 1–5
  performance_score?: number       // 0–10
  satisfaction?: number            // 1–5
  notes?: string
}

export interface FeedbackResponse extends FeedbackCreate {
  id: number
  created_at: string
}

// ── ML model status ──────────────────────────────────────────────────────────
export interface ModelStatus {
  yoga_image_classifier: boolean
  skeleton_classifier: boolean
  user_goal_classifier: boolean
  difficulty_predictor: boolean
}
