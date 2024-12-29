#FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
#FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

WORKDIR /app

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN apt update
RUN apt-get install fluidsynth -y

COPY ./src ./src
COPY ./app.py ./app.py
COPY test_scripts/test_tonal_plan.py ./test_tonal_plan.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
