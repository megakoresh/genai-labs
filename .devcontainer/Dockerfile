FROM mcr.microsoft.com/devcontainers/base:debian

ARG USER=vscode
ARG PYTHON_VERSION=3.12.13

# Install additional packages needed for builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    xz-utils \
    jq \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gdb lcov pkg-config \
    libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
    libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
    lzma liblzma-dev tk-dev uuid-dev zlib1g-dev libmpdec-dev libzstd-dev \
    inetutils-inetd

# Install Python 3.12 from source
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz \
    && tar -xf Python-${PYTHON_VERSION}.tar.xz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations --with-ensurepip=install --with-pydebug --with-lto \
    && make -j$(nproc) \
    && make altinstall \
    && cd .. \
    && rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tar.xz

RUN apt-get remove -y --no-install-recommends \
    build-essential gdb lcov pkg-config \
    libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
    libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
    lzma liblzma-dev tk-dev uuid-dev zlib1g-dev libmpdec-dev libzstd-dev \
    inetutils-inetd &&\
    apt-get autoremove -y --no-install-recommends && apt-get clean -y

USER ${USER}

# Persist bash history from WSL 2 host bind-mount
RUN SNIPPET="export HISTFILE=/tmp/.bash_history_persistent" \
    && echo "$SNIPPET" >> /home/${USER}/.bashrc \
    && echo "export HISTSIZE=10000" >> /home/${USER}/.bashrc \
    && echo "export HISTFILESIZE=20000" >> /home/${USER}/.bashrc \
    && echo "export HISTCONTROL=ignoredups:erasedups" >> /home/${USER}/.bashrc \
    && echo "shopt -s histappend" >> /home/${USER}/.bashrc \
    && touch /tmp/.bash_history_persistent \
    && chown ${USER}:${USER} /tmp/.bash_history_persistent

RUN git config --global core.excludesFile "/home/${USER}/.gitignore" && echo '.tmp' >> /home/${USER}/.gitignore
