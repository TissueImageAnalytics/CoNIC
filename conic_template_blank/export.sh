#!/usr/bin/env bash

./build.sh

docker save mitosisdetection | gzip -c > MitosisDetection.tar.gz
