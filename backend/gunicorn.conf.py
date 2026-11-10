import logging
import multiprocessing
import os

# Configure based on environment variables
ENV = os.environ.get("APP_ENV", "production").lower()
PORT = os.environ.get("BACKEND_PORT", "5001")
if PORT is None:
    logging.error("No port was specified for the backend service.")
    exit(1)

bind = f"0.0.0.0:{PORT}"

# Worker/threads defaults can be tuned with env vars
# max(2, multiprocessing.cpu_count() * 2 + 1)))
workers = int(os.environ.get("GUNICORN_WORKERS", 1))
threads = int(os.environ.get("GUNICORN_THREADS", 4))

# Only reload in non-production environments
reload = ENV != "production"

accesslog = "-"
errorlog = "-"

loglevel = os.environ.get("LOG_LEVEL", "info" if ENV ==
                          "production" else "debug")

# 4 minute timeout for worker because of long query times
timeout = int(os.environ.get("GUNICORN_TIMEOUT", 60 * 4))
