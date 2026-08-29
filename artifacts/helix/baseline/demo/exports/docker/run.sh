#!/usr/bin/env sh
set -eu
exec sloforge-gateway serve --config ./gateway.json
