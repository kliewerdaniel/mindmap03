#!/bin/bash
set -e

echo "entrypoint: Starting container as user $(id -u):$(id -g)"
echo "entrypoint: Running application (database will be created automatically)..."
exec "$@"
