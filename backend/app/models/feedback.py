from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text
from sqlalchemy.sql import func
from app.db.session import Base


class UserFeedback(Base):
    """
    Records user feedback on a recommendation item.

    completed         : 1 = completed, 0 = not completed, NULL = not reported
    perceived_difficulty: 1=very easy … 5=very hard
    comfort           : 1=uncomfortable … 5=very comfortable
    performance_score : 0–10 self-rated performance
    satisfaction      : 1–5 star rating
    notes             : free-text
    """
    __tablename__ = "user_feedback"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("user_profiles.id"), nullable=False)
    recommendation_id = Column(Integer, ForeignKey("recommendations.id"), nullable=True)
    category        = Column(String, nullable=False)   # yoga | fitness | ahar
    item_name       = Column(String, nullable=False)
    completed       = Column(Integer, nullable=True)   # 0 | 1
    perceived_difficulty = Column(Float, nullable=True)  # 1–5
    comfort         = Column(Float, nullable=True)       # 1–5
    performance_score    = Column(Float, nullable=True)  # 0–10
    satisfaction    = Column(Float, nullable=True)       # 1–5
    notes           = Column(Text, nullable=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
