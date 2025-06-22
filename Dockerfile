# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create directories with correct ownership
RUN mkdir -p /app/data /app/workspace

# Copy requirements file
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 consultant \
    && chown -R consultant:consultant /app

# Switch to non-root user
USER consultant

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH="/home/consultant/.local/bin:${PATH}"

# Default command
CMD ["python3"]
