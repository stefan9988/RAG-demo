# ---- Stage 1: Build ----
    FROM python:3.12-slim AS builder
    LABEL stage=builder
    
    # Set environment variables for Poetry
    ENV POETRY_VERSION=1.7.1 \
        POETRY_HOME="/opt/poetry" \
        POETRY_VIRTUALENVS_IN_PROJECT=true \
        POETRY_NO_INTERACTION=1 \
        PATH="/opt/poetry/bin:$PATH"
    
    # Install Poetry
    RUN apt-get update && apt-get install -y --no-install-recommends curl && \
        curl -sSL https://install.python-poetry.org | python3 - && \
        apt-get remove -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    
    # 1. Copy project definition, lock file, and source code
    COPY pyproject.toml poetry.lock ./
    COPY README.md ./ 
    COPY src/ ./src/
    
    # 2. Install dependencies using Poetry
    #    --sync ensures the environment exactly matches the lock file for these dependencies.
    #    --no-root is added here to explicitly tell poetry *only* to install dependencies
    #    and not try to install the current project itself in this step, as we'll do it with pip.
    #    (Alternatively, without --no-root, it *should* work, but we're being explicit now).
    RUN poetry install --no-dev --sync --no-root
    
    # 3. Install the current project (langchain-demo) itself using pip within Poetry's environment
    #    This will build `langchain-demo` from the `/app` context (using `pyproject.toml` and `src/`)
    #    and install it into the venv.
    RUN poetry run pip install .
    
    # ---- Stage 2: Runtime ----
    # (No changes to the runtime stage needed)
    FROM python:3.12-slim AS runtime
    
    ENV VIRTUAL_ENV=/app/.venv \
        PATH="/app/.venv/bin:$PATH" \
        PYTHONUNBUFFERED=1
    
    WORKDIR /app
    
    # Create a non-root user and switch to it
    RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser
    COPY --chown=appuser:appuser --from=builder /app /app
    USER appuser
    
    EXPOSE 8000
    
    CMD ["uvicorn", "langchain_demo.pipeline.api_main:app", "--host", "0.0.0.0", "--port", "8000"]