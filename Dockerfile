# Use the official TensorFlow Serving image as a base
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the Flask app code into the container
COPY . .

# Install the dependencies
RUN pip install -r requirements.txt

# Expose port 8080 for the Flask app
EXPOSE 8080

# Command to run the application
CMD ["python", "app.py"]