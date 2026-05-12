#!/bin/bash

# Usage: ./run.sh <num_threads> <dataset> <executable>
# dataset: "Tim" or "Adam"
# executable: "MultiQW", "InfQW" or "Anneal"

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <num_threads> <dataset> <executable>"
    echo "  dataset: Tim or Adam"
    echo "  executable: MultiQW, InfQW or Anneal"
    exit 1
fi

num_threads=$1
dataset=$2
executable=$3

binary="./${executable}"

# Configure dataset and executable specific parameters
data_dir="./data/${dataset}"
output_dir="./results/${dataset}_${executable}"

if [ "$dataset" == "Tim" ] && [ "$executable" == "MultiQW" ]; then
    n_values=(8 10 12 14 16 18)
    m_values=(1 2 5 10 20 50)
elif [ "$dataset" == "Tim" ] && [ "$executable" == "InfQW" ]; then
    n_values=(8 10)
    m_values=(1 2 5 10 20)
elif [ "$dataset" == "Tim" ] && [ "$executable" == "Anneal" ]; then
    n_values=(8 10 12 14 16 18)
    m_values=(1 2 5 10 20 50)
elif [ "$dataset" == "Adam" ] && [ "$executable" == "MultiQW" ]; then
    n_values=($(seq 5 18))
    m_values=(1 2 5 10 20)
elif [ "$dataset" == "Adam" ] && [ "$executable" == "InfQW" ]; then
    n_values=($(seq 5 10))
    m_values=(1 2 5 10 20)
elif [ "$dataset" == "Adam" ] && [ "$executable" == "Anneal" ]; then
    n_values=($(seq 5 18))
    m_values=(1 2 5 10 20)
else
    echo "Unknown dataset '$dataset'. Choose 'Tim' or 'Adam'."
    exit 1
fi

mkdir -p "$output_dir"

for n in "${n_values[@]}"; do
    filename="${data_dir}/SK_${n}n"

    if [ ! -f "$filename" ]; then
        echo "File not found, skipping: $filename"
        continue
    fi

    # Each problem consists of x*(x+1)/2 double precision values = 4*x*(x+1) bytes
    bytes_per_problem=$((4 * n * (n + 1)))
    file_size=$(stat -c%s "$filename")
    num_problems=$((file_size / bytes_per_problem))

    echo "Processing $filename: $num_problems problems"

    for m in "${m_values[@]}"; do

        base_problems_per_thread=$((num_problems / num_threads))
        extra_problems=$((num_problems % num_threads))

        output_file="${output_dir}/output_${n}_${m}"

        # Pre-allocate the file with the right number of bytes (4 bytes per float result)
        truncate -s $((num_problems * 4)) "$output_file"

        for (( thread=0; thread<num_threads; thread++ )); do
            start_point=$((thread * base_problems_per_thread + (thread < extra_problems ? thread : extra_problems)))
            problems_to_read=$((base_problems_per_thread + (thread < extra_problems ? 1 : 0)))

            echo "  n=$n, m=$m, thread=$thread: problems $start_point to $((start_point + problems_to_read - 1))"

            "$binary" "$n" "$m" "$filename" "$start_point" "$problems_to_read" "$output_dir" &
        done

        wait  # Wait for all threads of this m value to finish before moving on
    done
done