#FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
FROM ubuntu:22.04

# Update
RUN apt-get update && apt-get install fluidsynth python3-pip -y

RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

RUN apt-get update && apt-get install git -y

WORKDIR /app

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
