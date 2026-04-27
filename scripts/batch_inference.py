import base64
import json
import csv
from pathlib import Path
from openai import OpenAI

# ── Config ─────────────────────────────────────────────
IMAGES_DIR = Path("images/garage")
TRACKING_CSV = Path("data/garage_track_summary.csv")
OUTPUT_CSV = Path("results/garage_vlm_results.csv")
# ───────────────────────────────────────────────────────

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc1234"
)

PROMPT = """Analyze this vehicle image and provide the following details:
- Make
- Model
- Type (sedan, SUV, truck, coupe, van, motorcycle, etc.)
- Color

Respond in JSON format only, with keys: Make, Model, Type, Color.
Do not include any explanation or markdown, just the raw JSON object."""

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_image(image_path):
    b64 = encode_image(image_path)
    response = client.chat.completions.create(
        model="Qwen/Qwen2-VL-7B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    },
                    {
                        "type": "text",
                        "text": PROMPT
                    }
                ]
            }
        ],
        max_tokens=200
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

def find_image_for_track(track_id):
    """Find the best image for a track_id, preferring r1 then r2 then r3."""
    for rep in ["r1", "r2", "r3"]:
        pattern = f"track_{int(track_id):05d}_f*_{rep}.jpg"
        matches = list(IMAGES_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None

def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Load usable tracks from CSV
    tracks = []
    with open(TRACKING_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["usable_track"].strip() == "True":
                tracks.append(row)

    print(f"Found {len(tracks)} usable tracks")

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        fieldnames = ["track_id", "image_used", "Make", "Model", "Type", "Color", "error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, track in enumerate(tracks):
            track_id = track["track_id"]
            print(f"[{i+1}/{len(tracks)}] Track {track_id}...")

            img_path = find_image_for_track(track_id)

            if img_path is None:
                print(f"  -> No image found, skipping")
                writer.writerow({"track_id": track_id, "error": "no image found"})
                continue

            try:
                result = analyze_image(img_path)
                result["track_id"] = track_id
                result["image_used"] = img_path.name
                writer.writerow(result)
                print(f"  -> {result}")
            except Exception as e:
                print(f"  -> ERROR: {e}")
                writer.writerow({"track_id": track_id, "image_used": img_path.name, "error": str(e)})

    print(f"\nDone! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()