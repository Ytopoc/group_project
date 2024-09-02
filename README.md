# Завантажте застосунок і оберіть спосіб його запуску

# Локально
<ol>
  <li>Запускай файл <a href="https://github.com/Ytopoc/group_project/blob/main/download_models.py">download_models.py</a> щоб завантажити модель з хмари, або <a href="https://github.com/Ytopoc/group_project/blob/main/core.ipynb">core.ipynb</a> щоб самостійно навчити модель (це займає багато часу) </li>
  <li>Запускай файл<a href="https://github.com/Ytopoc/group_project/blob/main/main.py">main.py</a></li>
  <li><b>streamlit run main.py</b></li>
</ol>

# Dockerfile
<ol>
  <li><b>docker build -t app .</b></li>
  <li><b>docker run -it -p 8501:8501 app</b></li>
  <li>перейди за цим посиланням:http://localhost:8501</li>
</ol>

# docker-compose
<b>docker_compose up</b>

# Посилання на модель в гугл диску
https://drive.google.com/drive/u/1/folders/1hYDO3Dn8jJPnpkKtfkzG7Vb1RiAe8GKF
