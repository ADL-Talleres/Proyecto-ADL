from sqlalchemy import Column, Float, ForeignKey, String, Integer, DateTime, Enum, Boolean
from sqlalchemy.orm import relationship
import enum
import sys
sys.path.append('../')
from back.src.config.db_config import Base

class TaskStatus(enum.Enum):
    UPLOADED = 'UPLOADED'
    PROCESSED = 'PROCESSED'

class Task(Base):
    __tablename__ = 'tasks'

    id = Column(String, primary_key=True, index = True)
    name = Column(String)
    status = Column(Enum(TaskStatus), default=TaskStatus.UPLOADED)
    time_stamp = Column(DateTime)
    #segmentation = Column(List)
    input_path = Column(String)

    user_email = Column(String, ForeignKey("users.email"))

    user = relationship('User', back_populates='tasks')
