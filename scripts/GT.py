import os

root = "data/test_probe"
labels = {}

for label in ["text", "layout", "ocr", "mixed"]:
    folder = os.path.join(root, label)
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = f"{label}/{file}"
            labels[path] = label

# save
import json
with open("labels.json", "w") as f:
    json.dump(labels, f, indent=2)

print("Done!")
