# Toxic Comments Classifier

A Streamlit web app that classifies user-submitted comments into six categories of toxicity using a fine-tuned BERT model. The model is trained on the Jigsaw Toxic Comment dataset (multi-label).

Categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

## Stack

- PyTorch + HuggingFace Transformers (`bert-base-uncased`)
- Streamlit (web UI)
- Docker / docker-compose

## Files

| File | Purpose |
|------|---------|
| `core.ipynb` | End-to-end training notebook (preprocessing, training, evaluation) |
| `download_models.py` | Downloads pre-trained model weights from Google Drive |
| `main.py` | Streamlit app — loads the model and classifies user input |
| `Dockerfile` | Container image for the app |
| `docker-compose.yml` | Single-service compose file |

## Run locally

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Get the model weights (~ skip training)
python download_models.py
# OR train yourself by running core.ipynb (slow, needs GPU)

# 3. Launch the UI
streamlit run main.py
# http://localhost:8501
```

## Run with Docker

```bash
docker build -t toxic-classifier .
docker run -it -p 8501:8501 toxic-classifier
# http://localhost:8501
```

Or with docker-compose:

```bash
docker compose up -d
```

## Pre-trained model

If you don't want to retrain from scratch, the weights live on Google Drive:
https://drive.google.com/drive/u/1/folders/1hYDO3Dn8jJPnpkKtfkzG7Vb1RiAe8GKF

`download_models.py` pulls them into the local working directory.

## Authors

- [@Ytopoc](https://github.com/Ytopoc)
- [@Oleksiitaratynov](https://github.com/Oleksiitaratynov)
