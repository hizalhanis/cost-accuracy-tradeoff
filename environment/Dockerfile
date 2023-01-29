# hash:sha256:ddff9d34c9f2bbe34eca2e640a4ba16c91fd5974fc724c18e550b28e95520489
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        graphviz=2.42.2-3build2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    ipython==8.9.0 \
    matplotlib==3.6.3 \
    numpy==1.24.1 \
    pandas==1.5.3 \
    pydotplus==2.0.2 \
    scikit-learn==1.2.1 \
    seaborn==0.12.2
