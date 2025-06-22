FROM python:3.11-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user and set up directories
RUN useradd -m -u 1000 consultant && \
    mkdir -p /app/data /app/workspace /home/consultant/.local && \
    chown -R consultant:consultant /app && \
    chmod -R 755 /app && \
    # Allow pip installations for consultant user
    chown -R consultant:consultant /home/consultant/.local

# Switch to non-root user
USER consultant

# Ensure pip is up to date and install pip tools
RUN pip install --user --no-cache-dir --upgrade pip setuptools wheel

# Copy the execute script
COPY --chown=consultant:consultant execute.py /app/

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH="/home/consultant/.local/bin:${PATH}"

# Keep container running
CMD ["tail", "-f", "/dev/null"]
