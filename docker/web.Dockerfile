# The web tier and the stub worker. No CUDA, no torch, no diffusers.
#
# Keeping this image small is not tidiness -- it is what lets the always-on
# half of the product run on a $10 CPU box and start in seconds. The GPU
# worker is a separate image built from docker/Dockerfile.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies before source, so an edit to the code does not reinstall them.
COPY service/requirements.txt /app/service/requirements.txt
RUN pip install --no-cache-dir -r /app/service/requirements.txt

COPY service /app/service
COPY elements /app/elements

EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
