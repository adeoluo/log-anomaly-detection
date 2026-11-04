#!/usr/bin/env bash
# Download the public datasets used by the benchmarks (about 3.3 GB on disk).
#
#   scripts/download_data.sh [DEST]      (default: ../.bench-data/log-anomaly-detection)
#
# * Loghub HDFS_v1 and BGL (Zenodo record 8196385): raw logs for anomaly detection.
# * Loghub-2.0 (Zenodo record 8275861): 14 annotated parsing datasets, kept as zips and streamed.
# * Loghub-2.0 2k_dataset (GitHub): the 2,000-line annotated samples.
# * TCPD (GitHub): the Turing Change Point Dataset and its annotations.
# The datasets are not redistributed here; see each source for its license.
set -euo pipefail

DEST="${1:-../.bench-data/log-anomaly-detection}"
mkdir -p "$DEST" && cd "$DEST"

if [[ ! -f hdfs_v1/HDFS.log ]]; then
  curl -fL -o HDFS_v1.zip "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1"
  mkdir -p hdfs_v1 && unzip -q -o HDFS_v1.zip HDFS.log preprocessed/anomaly_label.csv -d hdfs_v1
  rm HDFS_v1.zip
fi

if [[ ! -f bgl_v1/BGL.log ]]; then
  curl -fL -o BGL.zip "https://zenodo.org/records/8196385/files/BGL.zip?download=1"
  mkdir -p bgl_v1 && unzip -q -o BGL.zip -d bgl_v1
  rm BGL.zip
fi

mkdir -p loghub2_full
for d in Apache Proxifier Linux Zookeeper Mac HealthApp Hadoop HPC OpenStack OpenSSH BGL Spark Thunderbird HDFS; do
  [[ -f "loghub2_full/$d.zip" ]] || curl -fL -o "loghub2_full/$d.zip" "https://zenodo.org/records/8275861/files/$d.zip?download=1"
done

if [[ ! -d loghub2/2k_dataset ]]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/logpai/loghub-2.0.git loghub2
  git -C loghub2 sparse-checkout set 2k_dataset
fi

if [[ ! -d tcpd ]]; then
  git clone -q --depth 1 https://github.com/alan-turing-institute/TCPD.git tcpd
  (cd tcpd && uv run --no-project --with-requirements requirements.txt python build_tcpd.py -v collect)
fi

du -sh "$PWD"
