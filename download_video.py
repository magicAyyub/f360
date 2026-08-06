import argparse
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests


def main():
    parser = argparse.ArgumentParser(description="Download a video from a URL.")
    parser.add_argument("url", help="The URL of the video to download.")
    args = parser.parse_args()

    url = args.url

    # Extract the real filename from the URL
    parsed_url = urlparse(url)
    filename = unquote(Path(parsed_url.path).name)
    if not filename:
        print("Error: Could not extract a valid filename from the URL.")
        return

    # Ensure the data directory exists
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://footballia.net/",
    }

    try:
        print(f"Downloading from: {url}")
        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=60,
        ) as response:
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with output_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue

                    file.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        progress = downloaded / total_size * 100
                        print(f"\rDownloading: {progress:.1f}%", end="", flush=True)

        print(f"\nSaved to: {output_path.resolve()}")

    except requests.RequestException as error:
        print(f"\nDownload failed: {error}")


if __name__ == "__main__":
    main()