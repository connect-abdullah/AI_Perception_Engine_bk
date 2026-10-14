# Next steps — after you paste your Supabase URIs into .env

# 1. Apply the schema
poetry run alembic upgrade head

# 2. Start FastAPI
poetry run uvicorn app.main:app --reload

# 3. (optional) Seed dummy data
poetry run python scripts/seed.py

# 4. Verify all eight tables
poetry run python scripts/verify_schema.py

# Offline tests (no DB needed)
poetry run pytest tests/ -v