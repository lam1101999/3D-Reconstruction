# Use the official Python image as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY requirements.txt .

RUN apt update 
RUN apt install cmake -y
# Install any dependencies
RUN pip install -r requirements.txt

# Install package.txt
COPY packages.txt .
RUN apt install libxrender1 procps libgl1-mesa-glx xvfb -y

# Copy source code
COPY . .
# Set the entry point for your app
CMD ["streamlit", "run", "src/main.py"]
EXPOSE 8501