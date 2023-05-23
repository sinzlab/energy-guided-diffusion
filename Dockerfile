FROM sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7 as base

# ADD .git to image to allow for commit hash retrieval
ADD . /src

WORKDIR /src

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade keyrings.alt
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir -e /src/lib/nnvision
RUN pip install --no-cache-dir wandb

RUN git clone https://github.com/sinzlab/mei
RUN python -m pip install --no-cache-dir -e /src/lib/mei

WORKDIR /src