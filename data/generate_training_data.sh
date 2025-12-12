#!/bin/bash

if [ ! -f "./scripts/concat.sh" ]; then
    echo "Error: 'concat.sh' not found"
    exit 1
fi

if [ ! -f "./scripts/inverse_concat.sh" ]; then
    echo "Error: 'inverse_concat.sh' not found"
    exit 1
fi

if [ ! -f "./scripts/concate_concats.sh" ]; then
    echo "Error: 'concate_concats.sh' not found"
    exit 1
fi

./scripts/concat.sh

./scripts/inverse_concat.sh

./scripts/concate_concats.sh
