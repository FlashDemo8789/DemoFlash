# FlashCAMP – 5-Minute Local Demo

## 1. Install & Train
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python notebooks/01_validate_csv.py --in data/camp_plus_balanced_with_meta.csv --schema backend/schema/camp_schema.yaml --out data/gold/v1.parquet
python pipelines/train_baseline.py --data data/gold/v1.parquet --models models/
python pipelines/gen_shap.py --data data/gold/v1.parquet --models models/ --out reports/assets/
```

## 2. Launch Services
```bash
uvicorn backend.app:app --port 8000

# In a new terminal
cd frontend
npm install
npm run dev     # http://localhost:3000

# In a new terminal (optional)
docker compose -f monitoring/docker-compose.yml up -d              # Grafana at http://localhost:3001
```

## 3. Use
* Open the React SPA & fill minimal fields ➜ pillar radar + probability.
* PDF report: `GET http://localhost:8000/report/<startup_id>` after prediction.
* Grafana shows live p99 latency & request throughput.

Happy demo! 