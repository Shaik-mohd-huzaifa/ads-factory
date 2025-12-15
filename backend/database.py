"""SQLite Database setup using SQLAlchemy"""
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = "sqlite:///./adaptive_brand.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Brand(Base):
    __tablename__ = "brands"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    primary_color = Column(String, default="#6366f1")
    secondary_color = Column(String, default="#8b5cf6")
    font_family = Column(String, default="Inter")
    tone = Column(String, default="professional")
    industry = Column(String, default="technology")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    assets = relationship("Asset", back_populates="brand", cascade="all, delete-orphan")
    campaigns = relationship("Campaign", back_populates="brand", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "primary_color": self.primary_color,
            "secondary_color": self.secondary_color,
            "font_family": self.font_family,
            "tone": self.tone,
            "industry": self.industry,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "assets": [asset.to_dict() for asset in self.assets]
        }


class Asset(Base):
    __tablename__ = "assets"
    
    id = Column(String, primary_key=True, index=True)
    brand_id = Column(String, ForeignKey("brands.id"), nullable=False)
    filename = Column(String, nullable=False)
    path = Column(String, nullable=False)
    content_type = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    brand = relationship("Brand", back_populates="assets")
    
    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "path": self.path,
            "content_type": self.content_type,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
        }


class Campaign(Base):
    __tablename__ = "campaigns"
    
    id = Column(String, primary_key=True, index=True)
    brand_id = Column(String, ForeignKey("brands.id"), nullable=False)
    prompt = Column(Text, nullable=False)
    brief = Column(JSON)  # Store brief as JSON
    master_design = Column(JSON)  # Store master design info as JSON
    creatives = Column(JSON)  # Store list of creatives as JSON
    relevant_assets = Column(JSON)  # Store relevant assets as JSON
    status = Column(String, default="completed")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    brand = relationship("Brand", back_populates="campaigns")
    
    def to_dict(self):
        return {
            "id": self.id,
            "brand_id": self.brand_id,
            "prompt": self.prompt,
            "brief": self.brief or {},
            "master_design": self.master_design or {},
            "creatives": self.creatives or [],
            "relevant_assets": self.relevant_assets or [],
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# Create all tables
def init_db():
    Base.metadata.create_all(bind=engine)


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
