FROM python:3.10

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables (modify as needed)
ENV PYTHONUNBUFFERED=1

# Default command to run experiments
CMD ["python", "losh.py", "loshfiles/finetuning.yml"]
