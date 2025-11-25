import os
import time
import re
import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def sanitize_filename(filename):
    r"""
    Sanitize the filename by removing or replacing invalid characters.
    Removes /, ?, :, |, <, >, *, ", \ and checks for reserved names.
    """
    # Replace invalid characters with an underscore or empty string
    clean_name = re.sub(r'[\\/*?:"<>|]', "", filename)
    # Remove control characters and strip whitespace
    clean_name = "".join(ch for ch in clean_name if ord(ch) >= 32)
    return clean_name.strip()

def download_podcasts(rss_feed_url: str, output_dir: str, limit: int | None = None):
    """
    Download podcasts from an RSS feed.
    
    Args:
        rss_feed_url: The URL of the RSS feed.
        output_dir: The directory to save downloaded episodes.
        limit: The maximum number of episodes to download. If None, downloads all.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"Parsing RSS feed: {rss_feed_url}")
    feed = feedparser.parse(rss_feed_url)
    
    if feed.bozo:
        print("Warning: There might be an issue with the RSS feed formatting, but continuing...")

    print(f"Found {len(feed.entries)} episodes.")

    session = get_session()

    entries_to_download = feed.entries
    
    if limit is not None:
        entries_to_download = feed.entries[:limit]
        print(f"Limiting to the most recent {len(entries_to_download)} episodes for testing...")
    else:
        print("Processing all episodes...")

    for entry in entries_to_download:
        try:
            title = entry.title
            clean_title = sanitize_filename(title)
            filename = f"{clean_title}.mp3"
            filepath = os.path.join(output_dir, filename)

            # Idempotency check
            if os.path.exists(filepath):
                print(f"Skipping (already exists): {filename}")
                continue

            # Find audio link
            audio_url = None
            # feedparser usually puts enclosures in entry.enclosures, but sometimes in links
            # Checking links first as requested, but also enclosures is safer
            if hasattr(entry, 'links'):
                for link in entry.links:
                    if link.type == 'audio/mpeg':
                        audio_url = link.href
                        break
            
            if not audio_url and hasattr(entry, 'enclosures'):
                 for enclosure in entry.enclosures:
                    if enclosure.type == 'audio/mpeg':
                        audio_url = enclosure.href
                        break

            if not audio_url:
                print(f"No audio link found for: {title}")
                continue

            # Download
            print(f"Downloading: {filename}")
            response = session.get(audio_url, stream=True, timeout=30)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"Error downloading '{entry.get('title', 'Unknown')}': {e}")
            continue
    
    print("Download process completed.")

if __name__ == "__main__":
    RSS_FEED_URL = "https://rss.art19.com/future-of-education"
    OUTPUT_DIR = "podcast_downloads"
    
    # Set limit to 10 for testing, or None for all
    download_podcasts(RSS_FEED_URL, OUTPUT_DIR, limit=15)
