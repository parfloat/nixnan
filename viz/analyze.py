import re
import json

# Parse Stage 2: Overflow timeline
stage2_file = "./traces/Stage2/baseline_native.txt"
overflow_data = {"FP16": None, "BF16": None, "FP32": None}

with open(stage2_file) as f:
    content = f.read()

# Find "first non-finite" lines
for fmt in ["FP16", "BF16", "FP32"]:
    pattern = f"===== {fmt}.*?first non-finite at step (\d+)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        step = int(match.group(1))
        overflow_data[fmt] = {"step": step, "time": step * 1e-3}

# Parse Stage 3: Exception counts
stage3_file = "./traces/Stage3/exceptions_summary.txt"
exception_data = {"FP16": {}, "BF16": {}, "FP32": {}}

with open(stage3_file) as f:
    lines = f.readlines()

current_fmt = None
for line in lines:
    if "--- FP16 Operations ---" in line:
        current_fmt = "FP16"
    elif "--- BF16 Operations ---" in line:
        current_fmt = "BF16"
    elif "--- FP32 Operations ---" in line:
        current_fmt = "FP32"
    elif "--- FP64" in line or "Memory" in line:
        current_fmt = None

    if current_fmt and line.strip():
        for exc_type in ["NaN", "Infinity", "Subnormal"]:
            if f"{exc_type}:" in line:
                parts = line.split()
                try:
                    count = int(parts[-3].rstrip('('))
                    repeats = int(parts[-1].rstrip(')'))
                    exception_data[current_fmt][exc_type] = {"count": count, "repeats": repeats}
                except:
                    pass

# Extract per-instruction errors
instr_data = {}
with open(stage3_file) as f:
    for line in f:
        if "error [" in line:
            # Extract format
            fmt = None
            if "type f16" in line:
                fmt = "FP16"
            elif "type bf16" in line:
                fmt = "BF16"
            elif "type f32" in line:
                fmt = "FP32"

            if fmt:
                # Extract instruction
                match = re.search(r'instruction\s+(\w+)', line)
                instr = match.group(1) if match else "OTHER"

                # Error type
                etype = "Infinity" if "infinity" in line else ("NaN" if "NaN" in line else "Other")

                key = (fmt, instr)
                if key not in instr_data:
                    instr_data[key] = 0
                instr_data[key] += 1

# Save as JSON for charting
data = {
    "overflow": overflow_data,
    "exceptions": exception_data,
    "instructions": instr_data
}

with open("./viz/data.json", "w") as f:
    json.dump(data, f, indent=2, default=str)

print("✓ Data extracted")
