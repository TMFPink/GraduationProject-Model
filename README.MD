python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
