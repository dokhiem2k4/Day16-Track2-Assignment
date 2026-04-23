#!/bin/bash
set -e

apt-get update -y
apt-get install -y python3 python3-pip python3-venv unzip curl

python3 -m pip install --upgrade pip
pip3 install lightgbm scikit-learn pandas numpy kaggle

mkdir -p /home/$(logname 2>/dev/null || echo ubuntu)/ml-benchmark
