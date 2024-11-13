import uuid
from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.security import HTTPAuthorizationCredentials
from fastapi_jwt_auth import AuthJWT
from sqlalchemy.orm import Session
import sys
import os
from fastapi import HTTPException
sys.path.append('../')

router = APIRouter(
    prefix="/files",
    tags=["files"],
    responses={404: {"detail": "Not found"}},
)


UPLOAD_DIR = "../back/uploads"  # Carpeta donde guardar los archivos
@router.post("/uploadfile", status_code=201)
async def upload_file(file: UploadFile = File(...)):
    try:
        # Leer el contenido del archivo
        contents = await file.read()

        # Verificar si la carpeta UPLOAD_DIR existe, si no, crearla
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

        extension = file.filename.split(".")[-1]
        id = str(uuid.uuid4())
        # Crear la ruta completa del archivo en la carpeta uploadsCopy
        file_path = os.path.join(UPLOAD_DIR, id + "." + extension)

        # Guardar el archivo en la carpeta uploadsCopy
        with open(file_path, "wb") as f:
            f.write(contents)

        return {"filename": id + "." + extension}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))