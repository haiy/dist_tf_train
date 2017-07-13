#!/usr/bin/env bash
kill -9 $(ps uax | grep "grpc_tensorflow_server.py" | awk '{print $2}')