FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY qqp ./qqp
COPY outputs/model.joblib ./outputs/model.joblib
# optional: bake in the question bank used by /search
COPY data ./data
EXPOSE 8000
CMD ["uvicorn", "qqp.api:app", "--host", "0.0.0.0", "--port", "8000"]
