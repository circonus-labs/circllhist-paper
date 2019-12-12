FROM jupyter/scipy-notebook

USER root

# Give $NB_USER passwordless sudo
RUN printf "$NB_USER\tALL=(ALL)\tNOPASSWD: ALL" > /etc/sudoers.d/$NB_USER

RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get install -y\
    autoconf \
    openjdk-11-jre

USER $NB_USER
RUN pip install --upgrade pip
RUN pip install pyjnius

# Download and install dependencies
RUN git clone https://github.com/circonus-labs/libcircllhist/ /tmp/circllhist
RUN git clone https://github.com/DataDog/sketches-py.git      /tmp/ddsketch
RUN git clone https://github.com/CamDavidsonPilon/tdigest.git /tmp/pytdigest
RUN git clone https://github.com/HdrHistogram/HdrHistogram_py /tmp/hdr

RUN cd /tmp/circllhist &&\
    git checkout 11211b79 &&\
    autoconf &&\
    ./configure --libdir /usr/lib/x86_64-linux-gnu/ &&\
    make && sudo make install &&\
    cd src/python && python3 setup.py install

RUN cd /tmp/ddsketch &&\
    git checkout aee723c &&\
    python3 setup.py install

RUN cd /tmp/pytdigest &&\
    git checkout 08acb0d &&\
    python3 setup.py install

RUN cd /tmp/hdr &&\
    git checkout d566355 &&\
    python3 setup.py install

# Install tdigest.jar
RUN cd /tmp/; wget http://repo1.maven.org/maven2/com/tdunning/t-digest/3.2/t-digest-3.2.jar -o tdigest.jar

EXPOSE 9998
EXPOSE 9999

ADD cmd.sh ./

CMD ["./cmd.sh"]
