# Prologue
# DO NOT CHANGE
from 812206152185.dkr.ecr.us-west-2.amazonaws.com/latch-base-cuda12:565f-main

workdir /tmp/docker-build/work/

shell [ \
    "/usr/bin/env", "bash", \
    "-o", "errexit", \
    "-o", "pipefail", \
    "-o", "nounset", \
    "-o", "verbose", \
    "-o", "errtrace", \
    "-O", "inherit_errexit", \
    "-O", "shift_verbose", \
    "-c" \
]
env TZ='Etc/UTC'
env LANG='en_US.UTF-8'
env PYTHONUNBUFFERED=1

arg DEBIAN_FRONTEND=noninteractive

# Latch SDK
# DO NOT REMOVE
run pip install latch==2.53.10
run mkdir /opt/latch

# Install Mambaforge
run apt-get update --yes && \
    apt-get install --yes curl git && \
    curl \
        --location \
        --fail \
        --remote-name \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    `# Docs for -b and -p flags: https://docs.anaconda.com/anaconda/install/silent-mode/#linux-macos` \
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda -u && \
    rm Miniforge3-Linux-x86_64.sh

# Set conda PATH
env PATH=/opt/conda/bin:$PATH
run conda config --set auto_activate_base false

# Build conda environment
copy environment.yml /opt/latch/environment.yaml
run mamba env create \
    --file /opt/latch/environment.yaml \
    --name spatialglue --yes
env PATH=/opt/conda/envs/spatialglue/bin:$PATH

# Install R/ArchR system libraries for ArchRProject coverage export.
# Keep R outside the conda env so mamba only solves the Python/CUDA stack.
run apt-get update --yes && \
    apt-get install --yes \
        apt-transport-https \
        aptitude \
        build-essential \
        default-jdk \
        gdebi-core \
        gfortran \
        libcairo2-dev \
        libatlas-base-dev \
        libbz2-dev \
        libcurl4-openssl-dev \
        libfontconfig1-dev \
        libfreetype6-dev \
        libfribidi-dev \
        libgdal-dev \
        libgit2-dev \
        libgsl-dev \
        libglpk-dev \
        libharfbuzz-dev \
        libhdf5-dev \
        libicu-dev \
        libjpeg-dev \
        liblzma-dev \
        libmagick++-dev \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libpcre3-dev \
        libssl-dev \
        libtcl8.6 \
        libtiff5 \
        libtiff-dev \
        libtk8.6 \
        libxml2-dev \
        libx11-dev \
        libxt-dev \
        locales \
        make \
        pandoc \
        r-base \
        r-base-dev \
        r-cran-rjava \
        tzdata \
        vim \
        wget \
        zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Use the known-good combined_cluster_wf R/ArchR lockfile.
run export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin && \
    unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH PKG_CONFIG_PATH CFLAGS CPPFLAGS LDFLAGS && \
    wget https://cran.r-project.org/src/base/R-4/R-4.3.3.tar.gz && \
    tar zxvf R-4.3.3.tar.gz && \
    rm R-4.3.3.tar.gz && \
    cd R-4.3.3 && \
    ./configure --enable-R-shlib && \
    make && \
    make install && \
    cd /tmp/docker-build/work && \
    rm -rf R-4.3.3
run /usr/local/bin/R CMD javareconf && \
    echo "TZ=$(cat /etc/timezone)" >> /etc/R/Renviron.site
run /usr/local/bin/R -e "install.packages('https://cran.r-project.org/src/contrib/Archive/renv/renv_1.0.7.tar.gz', repos = NULL, type = 'source')"
copy renv.lock /root/renv.lock
run mkdir -p /root/renv
copy renv/activate.R /root/renv/activate.R
copy renv/settings.json /root/renv/settings.json
workdir /root
run /usr/local/bin/R -e "renv::restore()" && \
    /usr/local/bin/Rscript -e "library(ArchR); cat('ArchR ', as.character(packageVersion('ArchR')), '\n', sep = '')"

run /usr/local/bin/Rscript -e "BiocManager::install('BSgenome.Mmusculus.UCSC.mm39', ask = FALSE, update = FALSE)" && \
    /usr/local/bin/Rscript -e "library(BSgenome.Mmusculus.UCSC.mm39); cat('BSgenome.Mmusculus.UCSC.mm39 loaded\n')"

workdir /tmp/docker-build/work/

# Install latch (pyflyte) inside the conda env so serialization sees workflow deps
run pip install latch==2.53.10
run pip install SpatialGlue==1.1.5
run pip install leidenalg==0.10.2
run pip install snapatac2

# Copy workflow data (use .dockerignore to skip files)
copy . /root/

# Latch workflow registration metadata
# DO NOT CHANGE
arg tag
# DO NOT CHANGE
env FLYTE_INTERNAL_IMAGE $tag

workdir /root
