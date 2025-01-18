#Specify the base image to user for the Docker image
#(Unify python and ubuntu environment by below script)
FROM python:3.12-slim

WORKDIR .

# Copy installed library doc to container from local machine
COPY requirements.txt .

# Copy files in local machine to container
COPY ./src .

# Run a command to install python library written in txt file
RUN pip install --no-cache-dir -r requirements.txt

#Specity the command  to run when the Docker container starts
#CMD ["python3"]
#keep container work 
CMD ["tail", "-f", "/dev/null"]

