#!/bin/bash
# Start tabpfn_compare.sh as soon as a Prior Labs API key is saved to ~/.cache/tabpfn/auth_token (checks every 30 s).
source $(dirname $0)/env.sh; cd $(dirname $0)
until [ -s $HOME/.cache/tabpfn/auth_token ] || [ -n "$TABPFN_TOKEN" ]; do sleep 30; done
echo "$(date +%F_%T) token found, starting queue"
exec ./tabpfn_compare.sh
