#!/usr/bin/env bash
TRAIN_PY_NAME=dist_train_example_model.py
kill -9 $(ps uax | grep "$TRAIN_PY_NAME" | awk '{print $2}' )