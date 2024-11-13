from sqlalchemy import Column, String
from sqlalchemy.orm import relationship
import sys
sys.path.append('../')
from back.src.config.db_config import Base

class User(Base):
    __tablename__ = 'users'

    email = Column(String, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)

    tasks = relationship('Task', back_populates='user', cascade='all, delete-orphan', passive_deletes=True)