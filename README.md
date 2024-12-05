# KidneySegNet

## Introducción

Este proyecto está estructurado en tres componentes principales, cada uno desempeñando un papel crucial en la funcionalidad general de la aplicación. A continuación, se presenta una visión general de cada componente:

1. **Backend (Carpeta: `back`)**
    - El backend de esta aplicación está desarrollado utilizando FastAPI, un framework web moderno y de alto rendimiento para construir APIs con Python 3.6+ basado en anotaciones de tipo estándar de Python. Este componente se encarga de manejar toda la lógica del lado del servidor, incluyendo los endpoints de la API, interacciones con la base de datos y procesamiento en el servidor.

2. **Frontend (Carpeta: `front`)**
    - El frontend está construido utilizando React, una biblioteca de JavaScript popular para construir interfaces de usuario. Esta parte de la aplicación maneja el renderizado del lado del cliente, proporcionando una experiencia de usuario dinámica e interactiva. Todos los componentes de la interfaz de usuario, gestión de estado y enrutamiento del lado del cliente se implementan aquí.

3. **Worker (Carpeta: `worker`)**
    - Este componente contiene los scripts necesarios para realizar tareas en segundo plano o procesamiento asíncrono. Puede incluir scripts para el procesamiento de datos, tareas programadas y otras operaciones que no requieren interacción directa con el usuario.

A continuación, se proporcionan instrucciones detalladas sobre cómo configurar y ejecutar cada parte del proyecto.

## Guía Paso a Paso para Ejecutar el Proyecto

### Prerrequisito: Clonar el Repositorio

1. **Clonar el repositorio en tu máquina local:**
   ```bash
   git clone https://github.com/ADL-Talleres/Proyecto-ADL.git
   cd Proyecto-ADL
   ```

### Backend (Carpeta: `back`)

1. **Navegar al directorio `back`:**
   ```bash
   cd back
   ```

2. **Crear un entorno virtual y activarlo:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
   ```

3. **Instalar las dependencias requeridas:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar el servidor de FastAPI:**
   ```bash
   uvicorn main:app --reload
   ```
   El servidor backend estará corriendo en `http://127.0.0.1:8000`.

### Frontend (Carpeta: `front`)

1. **Navegar al directorio `front`:**
   ```bash
   cd front
   ```

2. **Instalar las dependencias requeridas:**
   ```bash
   npm install
   ```

3. **Iniciar el servidor de desarrollo de React:**
   ```bash
   npm start
   ```
   El frontend estará corriendo en `http://localhost:3000`.

### Worker (Carpeta: `worker`)

1. **Navegar al directorio `worker`:**
   ```bash
   cd worker
   ```

2. **Crear un entorno virtual y activarlo:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
   ```

3. **Instalar las dependencias requeridas:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar el script principal del worker:**
   ```bash
   python main.py
   ```

### Despliegue con Docker (Opcional)

Para un despliegue containerizado, puedes utilizar el `Dockerfile` proporcionado para el backend y el frontend.

#### Backend y Worker

1. **Navegar al directorio raíz del proyecto (si no lo has hecho ya):**
   ```bash
   cd Proyecto-ADL
   ```

2. **Construir la imagen de Docker para el backend y el worker:**
   ```bash
   docker build -t proyecto_adl_backend_worker .
   ```

3. **Ejecutar el contenedor de Docker para el backend y el worker:**
   ```bash
   docker run -p 8000:8000 proyecto_adl_backend_worker
   ```

#### Frontend

1. **Navegar al directorio `front`:**
   ```bash
   cd front
   ```

2. **Construir la imagen de Docker para el frontend:**
   ```bash
   docker build -t proyecto_adl_front .
   ```

3. **Ejecutar el contenedor de Docker para el frontend:**
   ```bash
   docker run -p 3000:3000 proyecto_adl_front
   ```

Esto iniciará tanto el servicio de backend como el frontend dentro de contenedores de Docker. Asegúrate de que el frontend esté configurado para comunicarse con el backend a través de los endpoints de API apropiados.

Siguiendo estos pasos, deberías tener todo el proyecto en funcionamiento, permitiéndote interactuar con la API del backend, utilizar la interfaz del frontend y ejecutar las tareas del worker según sea necesario. 
