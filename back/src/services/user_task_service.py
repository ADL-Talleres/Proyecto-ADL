from sqlalchemy.orm import Session
from fastapi_jwt_auth import AuthJWT
import sys
sys.path.append('../')
from back.src.services.user_service import get_user_by_email
from back.src.models.user import User



def get_tasks_by_user_email(db: Session, Authorize: AuthJWT) -> list:
    email = Authorize.get_jwt_subject()
    user: User = get_user_by_email(db, email)
    return user.tasks