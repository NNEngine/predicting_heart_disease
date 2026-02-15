from pathlib import Path
import shutil

def main():
    input_path = Path("data/processed")
    output_path = Path("data/features")

    output_path.mkdir(parents=True, exist_ok=True)

    for file in input_path.glob("*.csv"):
        shutil.copy(file, output_path / file.name)

if __name__ == "__main__":
    main()
