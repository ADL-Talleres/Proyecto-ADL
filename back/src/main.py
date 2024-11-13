from fastapi import FastAPI, status, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from fastapi.staticfiles import StaticFiles
import sys
sys.path.append('../')
from back.src.config.settings import Settings
from back.src.routers import user, task, file
from back.src.config.db_config import engine, Base
from back.src.config.settings import AuthSettings



settings = Settings()
app = FastAPI(title=settings.PROJECT_NAME)

Base.metadata.create_all(bind=engine)

@AuthJWT.load_config
def get_config():
    return AuthSettings()

@app.exception_handler
def exception_handler(request: Request, exc: AuthJWTException):
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": exc.message}
    )

app.include_router(user.router)
app.include_router(task.router)
app.include_router(file.router)

app.mount("/uploads", StaticFiles(directory="./uploads"), name="uploads")
#app.mount("/uploads_reason", StaticFiles(directory="./uploads_reason"), name="uploads_reason")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def root():
    return RedirectResponse(url="/docs")