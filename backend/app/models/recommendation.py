from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.sql import func
from app.db.session import Base


class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user_profiles.id"), nullable=False)
    category = Column(String, nullable=False)   # yoga | fitness | ahar
    item_name = Column(String, nullable=False)
    description = Column(String)
    score = Column(Float, default=0.0)          # model confidence / relevance score
    # Adjusted score after user feedback; starts equal to score
    feedback_adjusted_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
