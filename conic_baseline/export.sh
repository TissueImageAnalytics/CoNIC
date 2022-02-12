#!/usr/bin/env bash

./build.sh

docker save conic-inference | gzip -c > conic-inference.tar.gz
