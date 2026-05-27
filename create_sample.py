import pandas as pd
import time

start_time = time.time()

print("=" * 60)
print("CREATING 100K SAMPLE DATASET")
print("=" * 60)

file_path = "data/raw/accepted_2007_to_2018Q4.csv"

chunksize = 50000
sample_size_per_chunk = 2000

sample_chunks = []

print("\nReading dataset in chunks...\n")

for i, chunk in enumerate(
    pd.read_csv(
        file_path,
        chunksize=chunksize,
        low_memory=False
    )
):
    print(f"Processing chunk {i + 1}...")

    sampled_chunk = chunk.sample(
        min(sample_size_per_chunk, len(chunk)),
        random_state=42
    )

    sample_chunks.append(sampled_chunk)

print("\nCombining sampled chunks...")

final_sample = pd.concat(sample_chunks)

print("Shuffling final dataset...")

final_sample = final_sample.sample(
    frac=1,
    random_state=42
)

print("Selecting final 100K rows...")

final_sample = final_sample.head(100000)

output_path = "data/raw/loan_sample_100k.csv"

final_sample.to_csv(
    output_path,
    index=False
)

end_time = time.time()

print("\n" + "=" * 60)
print("PROCESS COMPLETED")
print("=" * 60)

print(f"Final Shape: {final_sample.shape}")

print(f"Saved to: {output_path}")

print(f"Execution Time: {(end_time - start_time):.2f} seconds")