FROM sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7 as base

# ADD .git to image to allow for commit hash retrieval
ADD . /src

WORKDIR /src

RUN mkdir ./lib
RUN git clone -b model_builder https://github.com/sinzlab/nnvision.git ./lib/nnvision

RUN pip install --no-cache-dir --upgrade keyrings.alt
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir -e /src/lib/nnvision
