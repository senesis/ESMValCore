# To build this container, go to ESMValCore root folder and execute :
# docker build -t esmvalcore:latest . -f docker/Dockerfile
FROM bsces/miniconda3:debian-stretch-20190610

WORKDIR /src/ESMValCore
COPY environment.yml .
RUN conda update -y conda pip && conda env update --name base --file environment.yml && conda clean --all -y
COPY . .
RUN pip install . && pip cache purge

ENTRYPOINT ["esmvaltool"]
