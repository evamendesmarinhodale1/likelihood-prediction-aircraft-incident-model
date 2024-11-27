# Likelihood Prediction Aircraft Incident Model

## Overview
This project predicts the likelihood of aircraft incidents using machine learning models.

## Deployment
This project is deployed on [Render](https://render.com). Ensure the following:
- The start command in the Procfile is correctly set to `gunicorn flask_app:app`.
- Dependencies are listed in `requirements.txt`.
- Environment variables are set correctly in Render.

## Running Locally
1. Clone the repository:
   
git clone https://github.com/evamendesmarinhodale1/likelihood-prediction-aircraft-incident-model cd likelihood-prediction-aircraft-incident-model

2. Install dependencies:
pip install -r requirements.txt
markdown
3. Run the application:
python flask_app.py
markdown

## Repository Structure
- `flask_app.py`: Main application file.
- `requirements.txt`: List of Python dependencies.
- `Procfile`: Deployment instructions.
- `static/`: Static assets (e.g., CSS, JS).
- `templates/`: HTML templates.
