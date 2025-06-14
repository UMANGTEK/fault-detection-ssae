import os
import re
import pandas as pd

output_dir = "csv_outputs"
os.makedirs(output_dir, exist_ok=True)

pattern = re.compile(r"\s*\d+:\s*([01]+)\s+([01]+)")

for filename in os.listdir():
    if filename.endswith(".test"):
        circuit_name = filename.replace(".test", "")
        input_file = filename
        output_file = f"{output_dir}/{circuit_name}_dataset.csv"

        data = []
        with open(input_file, 'r') as file:
            for line in file:
                match = pattern.match(line)
                if match:
                    input_vec, output_vec = match.groups()
                    data.append([input_vec, output_vec, circuit_name])

        df = pd.DataFrame(data, columns=["input_vector", "output_vector", "circuit"])
        df.to_csv(output_file, index=False)
        print(f"✅ Parsed: {filename} → {output_file}")

print("🎉 All .test files converted to CSV.")

