from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
import sys
sys.path.append('../')
from back.src.models.task import TaskStatus

class TaskBase(BaseModel):
    name: str
    user_email: str
    input_path: str

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "name": "task name",
                "user_email": "user@outlook.com",
                "input_path": "path/to/file"
            }
        }

class TaskCreate(TaskBase):
    pass

class TaskRead(TaskBase):
    id: str
    status: TaskStatus
    time_stamp: datetime
    segmentation: List[List[int]]

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "name": "task name",
                "user_email": "user@outlook.com",
                "input_path": "path/to/file",
                "id": "4f21a77d-b8fa-47bb-8df6-b772a635bc19",
                "status": "UPLOADED",
                "time_stamp": "2021-07-07T00:00:00",
                "segmentation": [[0,1,1,0,0],[0,1,1,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
                
            }
        }