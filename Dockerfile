FROM python:3.11.3

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
WORKDIR $APP_HOME

# Copy dependencies first for caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
RUN playwright install --with-deps

# Copy the whole project
COPY . $APP_HOME

EXPOSE 8080

# Run FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]

