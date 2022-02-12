#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# ! If you want to change the name of the docker image,
# ! you must change accordingly the name in `test.sh`
# ! and in `build.sh`
docker build -t conic-inference "$SCRIPTPATH"
