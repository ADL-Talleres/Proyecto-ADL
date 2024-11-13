from fastapi_jwt_auth import AuthJWT
from fastapi import HTTPException

def authorized_user_email(Authorize: AuthJWT, user_email: str) -> str:
    print(Authorize.get_jwt_subject())
    authorized_user_email = Authorize.get_jwt_subject()
    if authorized_user_email != user_email:
        raise HTTPException(status_code=401, detail="Unauthorized access")