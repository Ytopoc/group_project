[English](README.md) | [**Українська**](README.uk.md)

# Класифікатор токсичних коментарів

Streamlit-вебзастосунок, що класифікує коментарі користувачів за шістьма категоріями токсичності, використовуючи fine-tuned модель BERT. Модель навчена на датасеті Jigsaw Toxic Comment (мультилейбл).

Категорії: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

## Стек

- PyTorch + HuggingFace Transformers (`bert-base-uncased`)
- Streamlit (веб-UI)
- Docker / docker-compose

## Файли

| Файл | Призначення |
|------|-------------|
| `core.ipynb` | End-to-end ноутбук тренування (препроцесинг, навчання, оцінка) |
| `download_models.py` | Завантажує ваги моделі з Google Drive |
| `main.py` | Streamlit-застосунок - завантажує модель і класифікує ввід |
| `Dockerfile` | Образ контейнера для застосунку |
| `docker-compose.yml` | Compose-файл з одним сервісом |

## Локальний запуск

```bash
# 1. Встановити залежності
pip install -r requirements.txt

# 2. Отримати ваги моделі (щоб не тренувати)
python download_models.py
# АБО натренувати самому, запустивши core.ipynb (повільно, потрібен GPU)

# 3. Запустити UI
streamlit run main.py
# http://localhost:8501
```

## Запуск через Docker

```bash
docker build -t toxic-classifier .
docker run -it -p 8501:8501 toxic-classifier
# http://localhost:8501
```

Або через docker-compose:

```bash
docker compose up -d
```

## Попередньо навчена модель

Якщо не хочете перенавчати з нуля - ваги лежать на Google Drive:
https://drive.google.com/drive/u/1/folders/1hYDO3Dn8jJPnpkKtfkzG7Vb1RiAe8GKF

`download_models.py` підтягує їх у локальну робочу директорію.

## Автори

- [@Ytopoc](https://github.com/Ytopoc)
- [@Oleksiitaratynov](https://github.com/Oleksiitaratynov)
