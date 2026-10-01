# Build the React front end.
FROM node:24-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Train the model and serve the API plus the built front end.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY transfer_value/ transfer_value/
# Training at build time means the model always matches the installed
# scikit-learn version; the raw data is deleted afterwards to keep the image small.
RUN python -m transfer_value.train && rm -rf data
COPY --from=frontend /app/frontend/dist frontend/dist
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn transfer_value.api:app --host 0.0.0.0 --port ${PORT}"]
