ARG PYTHON_VERSION=3.12
# workaround for https://github.com/moby/moby/issues/34482
FROM python:${PYTHON_VERSION}-slim AS python

FROM mcr.microsoft.com/devcontainers/base:debian

ARG BUN_VERSION=1.3.11
ARG GO_VERSION=1.25.8
ARG USER=vscode

RUN apt-get update && apt-get install -y --no-install-recommends \
    xz-utils jq wget bison mercurial unzip binutils gcc make bash-completion

# Install Python 3.12 from official prebuilt image
COPY --link --from=python /usr/local/ /usr/local/
RUN ldconfig

# Install Bun
RUN wget https://bun.com/install -O install_bun.sh \
    && chmod +x install_bun.sh \
    && BUN_INSTALL=/usr/local ./install_bun.sh "bun-v${BUN_VERSION}" \
    && rm install_bun.sh

# Install task
RUN sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin

# Install yq
RUN wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/local/bin/yq \
    && chmod +x /usr/local/bin/yq

USER ${USER}

# Install gvm, set default Go version and install upcloud cli
RUN bash -c "bash < <(curl -fsSL https://raw.githubusercontent.com/moovweb/gvm/master/binscripts/gvm-installer)" \
    && bash -c "source /home/${USER}/.gvm/scripts/gvm \
        && gvm install go${GO_VERSION} -B \
        && gvm use go${GO_VERSION} --default \
        && go install github.com/UpCloudLtd/upcloud-cli/v3/...@latest"

# Persist bash history from host so bash history is persisted across dev containers
RUN echo "export HISTSIZE=10000" >> /home/${USER}/.bashrc \
    && echo "export HISTFILESIZE=20000" >> /home/${USER}/.bashrc \
    && echo "export HISTCONTROL=ignoredups:erasedups" >> /home/${USER}/.bashrc \
    && echo "shopt -s histappend" >> /home/${USER}/.bashrc

RUN git config --global core.excludesFile "/home/${USER}/.gitignore" \
    && echo '.tmp' >> /home/${USER}/.gitignore \
    && echo '.venv' >> /home/${USER}/.gitignore
