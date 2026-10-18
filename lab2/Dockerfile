FROM python:3.12-alpine AS builder

WORKDIR /install
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-alpine AS runtime

ARG LAB_LOGIN=unknown
ARG LAB_TOKEN=unknown

LABEL org.lab.login="${LAB_LOGIN}" \
      org.lab.token="${LAB_TOKEN}"

ENV ROCKET_SIZE=Small \
    LAB_LOGIN="${LAB_LOGIN}" \
    LAB_TOKEN="${LAB_TOKEN}" \
    PORT=8000 \
    PYTHONUNBUFFERED=1

RUN adduser -D -u 10001 appuser

WORKDIR /app
COPY --from=builder /install /usr/local
COPY app/app.py /app/app.py

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8000/health || exit 1

CMD ["python", "app.py"]