FROM continuumio/miniconda3

ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create environment
COPY environment.yml .

RUN conda env create -f environment.yml

# Clone the yolov7 Git repository
# RUN git clone https://github.com/WongKinYiu/yolov7.git

# use the new environment - important
SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]

RUN conda install cython

RUN python -m pip install pip --upgrade

# Required for import cv2 error: `ImportError: libgthread-2.0.so.0: cannot open shared object file`
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libglib2.0-dev libgl1  -y

COPY ./requirements.txt /requirements.txt
# Install any needed packages specified in requirements.txt
RUN pip install -r /requirements.txt

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]

