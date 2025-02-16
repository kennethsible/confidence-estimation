FROM python:3.8-slim

# ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    g++ \
    make \
    wget \
    unzip \
    libboost-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/sillsdev/giza-py.git /giza-py
RUN pip install -r /giza-py/requirements.txt

ARG BOOST_DIR=/boost
ARG MGIZA_DIR=/mgiza

RUN wget https://archives.boost.io/release/1.86.0/source/boost_1_86_0.zip
RUN unzip boost_1_86_0.zip
RUN mv /boost_1_86_0 $BOOST_DIR
WORKDIR $BOOST_DIR
RUN ./bootstrap.sh --prefix=./build --with-libraries=thread,system
RUN ./b2 install

RUN git clone https://github.com/moses-smt/mgiza.git $MGIZA_DIR
WORKDIR $MGIZA_DIR/mgizapp
RUN cmake -DBOOST_ROOT=$BOOST_DIR/build -DBoost_USE_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX=/giza-py/.bin .
RUN make
RUN make install

WORKDIR /giza-py
