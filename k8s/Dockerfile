# syntax = docker/dockerfile:1.2
# I also maintain one with more personal bells/whistles here (direct use not recommended!):
# https://github.com/kingb12/docker-dev-env use as: `FROM kingb12/docker-dev-env`
FROM nvidia/cuda:11.8.0-base-ubuntu20.04


# default a user name. Change this if using
ARG user=bking2

# user id and group id, helpful to make these same as your host ones
ARG uid=501
ARG gid=20

# copy this to an environment variable https://blog.bitsrc.io/how-to-pass-environment-info-during-docker-builds-1f7c5566dd0e
ENV USER=${user}
ENV UID=${uid}
ENV GID=${gid}

# Remove any third-party apt sources to avoid issues with expiring keys. Then, install some basic utilities and python-dev
RUN rm -f /etc/apt/sources.list.d/*.list && \
    apt-get update && apt-get install -y \
    curl \
    zsh \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    wget \
    libx11-6 \
    python-dev \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /soe
WORKDIR /soe

# Create a non-root user and give it a home directory
RUN echo "User: ${USER}" && \
    groupadd -g ${GID} -o ${USER} && \
    useradd -u ${UID} -g ${GID} ${USER} && \
    echo "${USER}:${USER}" | chpasswd  && \
    mkdir -p /home/${USER} && chown -R ${USER}:${USER} /home/${USER}

# All users can use /home/${USER} as their home directory. Then, update permissions for .ssh to be more restrictive
ENV HOME=/home/${USER}
RUN mkdir $HOME/.cache $HOME/.config && \
    chmod -R 755 $HOME && \
    chmod -R 700 /home/${USER}/.ssh && \
    chmod 644 /home/${USER}/.ssh/authorized_keys


# work from the home directory
WORKDIR /home/${USER}

# Need to set user as owner of their home directory, now that we've populated things
RUN chown -R ${USER}:${USER} /home/${USER}

# Now continue with all actions as the non-privileged user
USER ${USER}

# Setup and install anaconda
RUN wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
    bash "Miniconda3-latest-Linux-x86_64.sh" -b -p "/home/${USER}/miniconda" && \
    miniconda/bin/conda init --all

# add this lint to the top of each shell file to turn off output in interactive modes
RUN echo "[[ "$-" != *i* ]] && return" > .bashrc && sed -i '1s/^/[[ "$-" != *i* ]] \&\& return\n/' .zshrc

# These can speed up builds that include conda/pip install (like below)
ENV PIP_CACHE_DIR .cache/buildkit/pip
RUN mkdir -p $PIP_CACHE_DIR

ENV PATH="/home/${USER}/miniconda/bin:${PATH}"

# create and enter the project directory
RUN mkdir hf_libraries_demo

WORKDIR "/home/${USER}/hf_libraries_demo"

# must run from repo root! Use .dockerignore to filter out things we don't want in the image
COPY --chown=$USER . .

# Set up a conda environment for this project and activate it and install torch and related libraries, then requirements
RUN --mount=type=cache,target=/opt/conda/pkgs conda init zsh && conda create python=3.10 --prefix venv

# conda init is needed before activation, but only takes effect after shell restart. Since each new Dockerfile line
# runs in a new shell instance, this doesn't actually take effect as expected. Instead, we'll change our shell to the
# one installed in the conda environment itself. Make RUN commands use the new environment:
SHELL ["conda", "run", "-p", "./venv", "/bin/bash", "-c"]

RUN --mount=type=cache,target=/opt/conda/pkgs conda install --yes --freeze-installed pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# this will install dependencies specified in setup.cfg, verified with command
RUN --mount=type=cache,target=.cache/buildkit/pip pip install --user -e . && \
    echo -e "import hf_libraries_demo.package_demo.addition_module as mymod\nprint(f'4 + 5 is {mymod.add_five_to_x(4)}')" | python
