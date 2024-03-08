#!/bin/bash
# set -x

cat <<EOF
Starting LangFlow Server...
EOF

langflow run --host 0.0.0.0 --port 8080
