"""
ml/recommendation/yoga_metadata.py

Structured yoga pose metadata table.

Each entry is a fully-specified pose with feature attributes used by the
recommendation engine to compute suitability scores.

Source: Based on widely available yoga reference literature (B.K.S. Iyengar,
  Yoga Journal, standard yoga teacher training curricula).
  This is curated domain knowledge, not ML output.

Fields:
  name              : canonical pose name
  sanskrit_name     : Sanskrit name (if distinct)
  category          : [standing, seated, supine, prone, inversion, balancing, twisting, backbend]
  difficulty        : Beginner | Intermediate | Advanced (1/2/3 numeric)
  target_areas      : list of body areas
  flexibility_req   : minimum flexibility score required (1–5)
  strength_req      : minimum strength score required (1–5)
  experience_req    : minimum yoga experience (0=none, 1=beginner, 2=intermediate, 3=advanced)
  duration_min      : recommended hold / practice duration in minutes
  goals             : list of wellness goals this pose serves
  benefits          : brief description of benefits
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import json


@dataclass
class YogaPose:
    name: str
    sanskrit_name: str
    category: str
    difficulty: str                  # Beginner | Intermediate | Advanced
    difficulty_num: int              # 1 | 2 | 3
    target_areas: list[str]
    flexibility_req: float           # 1–5
    strength_req: float              # 1–5
    experience_req: int              # 0–3
    duration_min: float
    goals: list[str]                 # matches GOAL_CLASSES
    benefits: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Pose catalog (82 poses aligned with Yoga-82 classes where possible)
# Only a representative subset is hardcoded here; expand as needed.
# ─────────────────────────────────────────────────────────────────────────────

YOGA_POSES: list[YogaPose] = [
    # ── Standing Poses ────────────────────────────────────────────────────────
    YogaPose("Mountain Pose",       "Tadasana",         "standing",  "Beginner",     1, ["posture","legs","core"],      1.0, 1.0, 0, 2.0,  ["General Fitness","Mobility","Stress Reduction"],       "Improves posture, grounding, body awareness"),
    YogaPose("Warrior I",           "Virabhadrasana I", "standing",  "Beginner",     1, ["hips","legs","shoulders"],    2.0, 2.0, 0, 3.0,  ["Strength","General Fitness","Weight Management"],      "Builds leg and core strength, opens chest"),
    YogaPose("Warrior II",          "Virabhadrasana II","standing",  "Beginner",     1, ["hips","legs","arms"],         2.0, 2.0, 0, 3.0,  ["Strength","General Fitness","Weight Management"],      "Strengthens legs, improves stamina and focus"),
    YogaPose("Warrior III",         "Virabhadrasana III","standing", "Intermediate", 2, ["core","legs","glutes"],       2.0, 3.0, 1, 2.0,  ["Strength","Mobility","General Fitness"],               "Full-body strengthening, improves balance"),
    YogaPose("Triangle Pose",       "Trikonasana",      "standing",  "Beginner",     1, ["hips","hamstrings","spine"],  3.0, 1.0, 0, 3.0,  ["Flexibility","Mobility","General Fitness"],            "Stretches hips, opens thoracic spine"),
    YogaPose("Tree Pose",           "Vrksasana",        "balancing", "Beginner",     1, ["legs","core","ankles"],       2.0, 2.0, 0, 3.0,  ["Mobility","Stress Reduction","General Fitness"],       "Improves balance and concentration"),
    YogaPose("Extended Side Angle", "Utthita Parsvakonasana","standing","Beginner",  1, ["hips","legs","obliques"],     2.0, 2.0, 0, 3.0,  ["Flexibility","Strength","General Fitness"],            "Stretches side body, builds stamina"),
    YogaPose("Chair Pose",          "Utkatasana",       "standing",  "Beginner",     1, ["quads","glutes","core"],      1.0, 3.0, 0, 2.0,  ["Strength","Weight Management","General Fitness"],      "Builds lower body strength and endurance"),
    YogaPose("Half Moon Pose",      "Ardha Chandrasana","balancing", "Intermediate", 2, ["hips","legs","core"],         3.0, 3.0, 1, 2.0,  ["Mobility","Flexibility","General Fitness"],            "Strengthens legs, improves coordination"),
    YogaPose("Eagle Pose",          "Garudasana",       "balancing", "Intermediate", 2, ["shoulders","hips","ankles"],  3.0, 2.0, 1, 2.0,  ["Mobility","Stress Reduction","Flexibility"],           "Releases shoulder/hip tension, improves focus"),

    # ── Seated Poses ──────────────────────────────────────────────────────────
    YogaPose("Staff Pose",          "Dandasana",        "seated",    "Beginner",     1, ["spine","hamstrings","core"],  2.0, 2.0, 0, 2.0,  ["Mobility","Flexibility","General Fitness"],            "Strengthens spine and improves posture"),
    YogaPose("Seated Forward Bend", "Paschimottanasana","seated",    "Beginner",     1, ["hamstrings","spine","calves"],3.0, 1.0, 0, 3.0,  ["Flexibility","Stress Reduction","Mobility"],           "Deep hamstring stretch, calms nervous system"),
    YogaPose("Butterfly Pose",      "Baddha Konasana",  "seated",    "Beginner",     1, ["hips","groin","adductors"],  3.0, 1.0, 0, 3.0,  ["Flexibility","Mobility","Stress Reduction"],           "Opens hips, relieves lower-back tension"),
    YogaPose("Half Pigeon",         "Eka Pada Rajakapotasana","seated","Intermediate",2, ["hips","IT band","piriformis"],4.0,1.0, 1, 4.0,  ["Flexibility","Mobility","Stress Reduction"],           "Intense hip opener, releases deep hip tension"),
    YogaPose("Full Pigeon",         "Kapotasana",       "backbend",  "Advanced",     3, ["hip flexors","spine","chest"],5.0, 3.0, 3, 3.0,  ["Flexibility","Mobility"],                              "Advanced backbend, full hip flexor opening"),
    YogaPose("Lotus Pose",          "Padmasana",        "seated",    "Intermediate", 2, ["hips","knees","ankles"],     4.0, 1.0, 1, 5.0,  ["Stress Reduction","Flexibility","General Fitness"],    "Meditative seat, deep hip rotation"),
    YogaPose("Cow Face Pose",       "Gomukhasana",      "seated",    "Intermediate", 2, ["shoulders","hips","arms"],   3.0, 1.0, 1, 3.0,  ["Flexibility","Mobility","Stress Reduction"],           "Deep shoulder and hip stretch"),
    YogaPose("Boat Pose",           "Navasana",         "seated",    "Intermediate", 2, ["core","hip flexors","spine"],2.0, 4.0, 1, 2.0,  ["Strength","Weight Management","General Fitness"],      "Core strengthener, improves balance"),
    YogaPose("Head-to-Knee Pose",   "Janu Sirsasana",   "seated",    "Beginner",     1, ["hamstrings","hips","spine"], 3.0, 1.0, 0, 3.0,  ["Flexibility","Mobility","Stress Reduction"],           "Hamstring and spine stretch with a gentle twist"),

    # ── Supine Poses ──────────────────────────────────────────────────────────
    YogaPose("Corpse Pose",         "Savasana",         "supine",    "Beginner",     1, ["whole body"],                1.0, 1.0, 0, 5.0,  ["Stress Reduction","General Fitness"],                  "Complete relaxation, integration of practice"),
    YogaPose("Happy Baby",          "Ananda Balasana",  "supine",    "Beginner",     1, ["hips","lower back","groin"], 2.0, 1.0, 0, 3.0,  ["Flexibility","Stress Reduction","Mobility"],           "Gently stretches inner thighs and lower back"),
    YogaPose("Supine Twist",        "Supta Matsyendrasana","supine",  "Beginner",    1, ["spine","shoulders","hips"],  2.0, 1.0, 0, 3.0,  ["Mobility","Stress Reduction","Flexibility"],           "Spinal rotation, releases lower back"),
    YogaPose("Bridge Pose",         "Setu Bandhasana",  "supine",    "Beginner",     1, ["glutes","spine","chest"],    2.0, 2.0, 0, 3.0,  ["Strength","Mobility","General Fitness"],               "Strengthens glutes and spine, opens chest"),
    YogaPose("Legs Up The Wall",    "Viparita Karani",  "inversion", "Beginner",     1, ["legs","lower back"],         1.0, 1.0, 0, 5.0,  ["Stress Reduction","Mobility","General Fitness"],       "Improves circulation, deeply restorative"),
    YogaPose("Wind Relieving Pose", "Pawanmuktasana",   "supine",    "Beginner",     1, ["lower back","hips"],         1.0, 1.0, 0, 2.0,  ["Mobility","Stress Reduction","Flexibility"],           "Releases lower-back tension, aids digestion"),

    # ── Prone Poses ───────────────────────────────────────────────────────────
    YogaPose("Cobra Pose",          "Bhujangasana",     "prone",     "Beginner",     1, ["spine","chest","shoulders"], 2.0, 2.0, 0, 2.0,  ["Flexibility","Strength","General Fitness"],            "Opens chest, strengthens spine"),
    YogaPose("Sphinx Pose",         "Salamba Bhujangasana","prone",  "Beginner",     1, ["spine","chest"],             1.0, 1.0, 0, 3.0,  ["Flexibility","Mobility","Stress Reduction"],           "Gentle backbend, good for beginners"),
    YogaPose("Locust Pose",         "Salabhasana",      "prone",     "Beginner",     1, ["back muscles","glutes","legs"],1.0,3.0, 0, 2.0,  ["Strength","General Fitness","Weight Management"],      "Strengthens entire posterior chain"),
    YogaPose("Bow Pose",            "Dhanurasana",      "prone",     "Intermediate", 2, ["spine","chest","quads"],     3.0, 2.0, 1, 2.0,  ["Flexibility","Strength","General Fitness"],            "Full backbend, opens chest and hip flexors"),
    YogaPose("Full Bow",            "Purna Dhanurasana","prone",     "Advanced",     3, ["spine","chest","shoulders"], 5.0, 3.0, 3, 1.5,  ["Flexibility"],                                         "Extreme backbend requiring advanced flexibility"),

    # ── Inversions ────────────────────────────────────────────────────────────
    YogaPose("Downward Dog",        "Adho Mukha Svanasana","inversion","Beginner",  1, ["hamstrings","calves","shoulders","spine"],2.0,2.0,0, 3.0, ["General Fitness","Flexibility","Weight Management"], "Full-body stretch, mild inversion benefits"),
    YogaPose("Shoulder Stand",      "Sarvangasana",     "inversion", "Intermediate",2, ["neck","shoulders","core"],    2.0, 3.0, 1, 3.0,  ["Stress Reduction","Mobility","General Fitness"],       "Full inversion, improves circulation"),
    YogaPose("Headstand",           "Sirsasana",        "inversion", "Advanced",    3, ["core","shoulders","neck"],    2.0, 4.0, 2, 2.0,  ["Strength","Mobility","General Fitness"],               "King of asanas — builds focus and upper body strength"),
    YogaPose("Handstand",           "Adho Mukha Vrksasana","inversion","Advanced",  3, ["shoulders","core","wrists"],  2.0, 5.0, 3, 1.0,  ["Strength","General Fitness"],                          "Full body strength and balance challenge"),
    YogaPose("Forearm Stand",       "Pincha Mayurasana","inversion",  "Advanced",   3, ["shoulders","core","back"],    3.0, 5.0, 3, 1.0,  ["Strength","Flexibility"],                              "Builds shoulder and core strength"),

    # ── Sun Salutation / Flow ─────────────────────────────────────────────────
    YogaPose("Sun Salutation A",    "Surya Namaskar A", "standing",  "Beginner",    1, ["whole body"],                 2.0, 2.0, 0, 5.0,  ["Weight Management","General Fitness","Strength"],      "Dynamic full-body warm-up, cardiovascular"),
    YogaPose("Sun Salutation B",    "Surya Namaskar B", "standing",  "Intermediate",2, ["whole body"],                 2.0, 3.0, 1, 8.0,  ["Weight Management","Strength","General Fitness"],      "More vigorous flow with Warrior I integration"),

    # ── Twists ────────────────────────────────────────────────────────────────
    YogaPose("Seated Spinal Twist", "Ardha Matsyendrasana","seated", "Beginner",    1, ["spine","shoulders","hips"],   2.0, 1.0, 0, 3.0,  ["Mobility","Flexibility","Stress Reduction"],           "Detoxifying spinal rotation, opens shoulders"),
    YogaPose("Revolved Triangle",   "Parivrtta Trikonasana","standing","Intermediate",2,["spine","hips","hamstrings"],  3.0, 2.0, 1, 2.0,  ["Mobility","Flexibility","General Fitness"],            "Deep twist with balance challenge"),
    YogaPose("Revolved Chair",      "Parivrtta Utkatasana","standing","Intermediate",2,["spine","quads","core"],       2.0, 3.0, 1, 2.0,  ["Strength","Mobility","General Fitness"],               "Strengthening twist, good for core"),

    # ── Backbends ────────────────────────────────────────────────────────────
    YogaPose("Camel Pose",          "Ustrasana",        "backbend",  "Intermediate",2, ["spine","chest","hip flexors"],3.0, 2.0, 1, 2.0,  ["Flexibility","Mobility","General Fitness"],            "Deep backbend, stretches anterior chain"),
    YogaPose("Wheel Pose",          "Urdhva Dhanurasana","backbend", "Advanced",    3, ["spine","chest","shoulders","wrists"],4.0,4.0,2, 1.5, ["Flexibility","Strength"],                           "Full backbend, requires significant strength and flexibility"),
    YogaPose("Fish Pose",           "Matsyasana",       "backbend",  "Beginner",    1, ["chest","throat","spine"],     2.0, 1.0, 0, 3.0,  ["Flexibility","Stress Reduction","General Fitness"],    "Heart opener, relieves neck tension"),

    # ── Balancing ────────────────────────────────────────────────────────────
    YogaPose("Warrior III Balance", "Virabhadrasana III","balancing","Intermediate",2, ["core","legs","glutes"],       2.0, 3.0, 1, 2.0,  ["Strength","Mobility","General Fitness"],               "Full-body balance and strength"),
    YogaPose("Crow Pose",           "Bakasana",         "balancing", "Intermediate",2, ["wrists","core","shoulders"],  2.0, 4.0, 2, 1.5,  ["Strength","General Fitness"],                          "Arm balance builds wrist and core strength"),
    YogaPose("Crane Pose",          "Kakasana",         "balancing", "Intermediate",2, ["wrists","core","arms"],       2.0, 4.0, 2, 1.5,  ["Strength","General Fitness"],                          "Arm balance, builds focus and upper body strength"),
    YogaPose("Side Crow",           "Parsva Bakasana",  "balancing", "Advanced",    3, ["core","arms","obliques"],     2.0, 4.0, 2, 1.0,  ["Strength","General Fitness"],                          "Twisting arm balance, advanced core challenge"),

    # ── Restorative ──────────────────────────────────────────────────────────
    YogaPose("Child's Pose",        "Balasana",         "prone",     "Beginner",    1, ["lower back","hips","shoulders"],2.0,1.0, 0, 5.0, ["Stress Reduction","Mobility","Flexibility"],           "Deeply restorative, relieves stress and fatigue"),
    YogaPose("Reclined Hero",       "Supta Virasana",   "supine",    "Intermediate",2, ["quads","hip flexors","ankles"],4.0,1.0, 1, 3.0,  ["Flexibility","Mobility"],                              "Intense quad and ankle stretch"),
    YogaPose("Yin Forward Fold",    "Yin Uttanasana",   "standing",  "Beginner",    1, ["hamstrings","calves","spine"],3.0, 1.0, 0, 5.0,  ["Flexibility","Stress Reduction","Mobility"],           "Long-hold forward fold for connective tissue release"),
]

# Build lookup by name
POSE_BY_NAME: dict[str, YogaPose] = {p.name: p for p in YOGA_POSES}


def get_pose(name: str) -> YogaPose | None:
    return POSE_BY_NAME.get(name)


def get_all_poses() -> list[YogaPose]:
    return YOGA_POSES


def poses_to_dataframe():
    import pandas as pd
    return pd.DataFrame([p.to_dict() for p in YOGA_POSES])
