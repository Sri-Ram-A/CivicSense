# CivicSense
This project is about performing Sentiment Analysis on various Social Reforms and display insights on these sentiments . The final goal is making an web extension with the main focus on MLOPS
- Lets go

.
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── src/
│   ├── data_processing/
│   │   └── __init__.py
│   │   └── etl.py
│   ├── feature_engineering/
│   │   └── __init__.py
│   │   └── features.py
│   ├── models/
│   │   └── __init__.py
│   │   └── train.py
│   │   └── predict.py
│   ├── utils/
│   │   └── __init__.py
│   │   └── helpers.py
│   └── app.py  # For model serving (e.g., FastAPI)
├── notebooks/
│   ├── exploration.ipynb
│   └── model_experimentation.ipynb
├── tests/
│   ├── unit/
│   └── integration/
├── models/  # Stored trained models (e.g., serialized files)
├── config/
│   └── config.yaml
├── mlruns/  # MLflow tracking artifacts
├── docs/
├── .env
├── Makefile
├── requirements.txt
├── Dockerfile
├── README.md