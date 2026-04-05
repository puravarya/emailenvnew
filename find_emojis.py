import os

# Start from current folder
folder_path = os.path.dirname(__file__)

# Walk through all folders and files
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, start=1):
                    # Check for any non-ASCII character (emoji included)
                    if any(ord(c) > 127 for c in line):
                        print(f"{file}:{i}: {line.strip()}")
        except Exception as e:
            print(f"Could not read {file_path}: {e}")