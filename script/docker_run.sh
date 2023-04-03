#!/bin/bash
docker run --name santazip -p 60001:60001 -p 60002:60002 -p 60003:60003 --gpus all -it -v $(pwd):/santazip --rm tensorflow/tensorflow:2.11.0-gpu
