# Use Python 3.10 instead of 3.9
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy application files
COPY . .

# Install dependencies
RUN pip install --upgrade pip  # Upgrade pip first
RUN pip install -r requirements.txt

ENV PORT 8080

# Expose port (if needed)
EXPOSE 8080

# Run the application
CMD ["python", "app.py","0.0.0.0:8080"]
