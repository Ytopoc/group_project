# core.ipynb заванантажувати не потрібно, всі моделі завантажуються звідси: 
# https://drive.google.com/drive/u/1/folders/1hYDO3Dn8jJPnpkKtfkzG7Vb1RiAe8GKF

# group_project
# Dockerfile
docker build -t app .<br/>
docker run -it -p 8501:8501 app<br/>
streamlit run main.py

# docker-compose
docker_compose up