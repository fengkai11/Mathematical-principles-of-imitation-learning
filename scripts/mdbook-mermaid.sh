#!/usr/bin/env sh
set -eu

if [ -x /home/fengkai/.cargo/bin/mdbook-mermaid ]; then
  exec /home/fengkai/.cargo/bin/mdbook-mermaid "$@"
fi

exec mdbook-mermaid "$@"
