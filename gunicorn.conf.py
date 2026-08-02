"""Single-process gevent configuration required by the in-memory SSE event bus."""
from agent.settings import load_settings

settings = load_settings()

bind = f"{settings.host}:{settings.port}"
worker_class = "gevent"
workers = 1
timeout = 0
graceful_timeout = 30
accesslog = "-"
errorlog = "-"
