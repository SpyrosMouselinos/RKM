# Use the official Pytorch image from the Docker Hub
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set the working directory in the container
WORKDIR /signals

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY ./app /signals/app

# Expose the port the app runs on
EXPOSE 8000

# Change to working directory
WORKDIR /signals/app

# Command to run the application using uvicorn
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]