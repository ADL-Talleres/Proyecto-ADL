from fastapi import APIRouter, Depends, Body
from fastapi.security import HTTPBearer
from fastapi_jwt_auth import AuthJWT
from sqlalchemy.orm import Session
import sys
sys.path.append('../')
from back.src.schemas.user import UserCreate, UserLogin
import back.src.services.user_service as service
from back.src.config.db_config import get_db


router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"detail": "Not found"}},
)

get_bearer_token = HTTPBearer(auto_error=False)

@router.post("/", response_model=UserCreate, status_code=201)
def create_user(user: UserCreate = Body(...), db: Session = Depends(get_db)):
    return service.create_user(db, user)


@router.post("/login", status_code=201)
def login_user(user: UserLogin, Authorize: AuthJWT = Depends(), db: Session = Depends(get_db)):
    db_user = service.login_user(db, user)
    access_token = Authorize.create_access_token(subject=db_user.email, expires_time=7200)
    return {"access_token": access_token, "token_type": "bearer", "username": db_user.username, "email": db_user.email, "password": db_user.password}