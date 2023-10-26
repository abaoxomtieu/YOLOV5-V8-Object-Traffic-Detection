run_api:
	uvicorn api:api --port 8000

run_app:
	streamlit run app.py --server.fileWatcherType=none

