#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

#! replace `conic-inference` accordingly in `test.sh`
docker build -t conic-inference "$SCRIPTPATH"
