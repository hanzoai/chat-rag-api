# Pin the Debian suite (bookworm) so the apt package set is stable — the bare
# python:3.10 tag floated onto a release where libgl1-mesa-glx was removed.
FROM python:3.10-bookworm AS main

WORKDIR /app

# Install pandoc and netcat. libgl1 (bookworm) replaces the old
# libgl1-mesa-glx and provides the OpenGL runtime opencv/unstructured need.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    pandoc \
    netcat-openbsd \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download standard NLTK data, to prevent unstructured from downloading packages at runtime
RUN python -m nltk.downloader -d /app/nltk_data punkt_tab averaged_perceptron_tagger
ENV NLTK_DATA=/app/nltk_data

# Disable Unstructured analytics
ENV SCARF_NO_ANALYTICS=true

COPY . .

CMD ["python", "main.py"]
