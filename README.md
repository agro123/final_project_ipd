# final_project_ipd

**Descripción:**
Proyecto final Infraestructuras paralelas y distribuidas.
Tiene como objetivo demostrar la implementación, despliegue y
monitoreo de pipelines de aprendizaje automático en entornos distribuidos, utilizando servicios
de la nube como AWS.

**Cómo ejecutar:**

- **Build image:**
   ```bash
   docker build -t movies_revenue .


- **Run container:**
   ```bash
      docker run -it --name mr_c --rm -p 5000:5000 -p 5001:5001 -p 8888:8888 -v "$(pwd)/pip":/pip  -v "$(pwd)/data":/app/data -v "$(pwd)/model":/app/model -v "$(pwd)/notebook":/app/ -v "$(pwd)/models":/app/models movies_revenue
