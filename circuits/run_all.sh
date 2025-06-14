#!/bin/bash

mkdir -p results

for file in *.bench; do
    base=$(basename "$file" .bench)
    echo "Running ATALANTA on $file..."
    atalanta "$file" > "results/$base.test"
done

echo "✅ All .bench files processed. Output saved in results/"

