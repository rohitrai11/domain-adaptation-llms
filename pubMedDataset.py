# This script downloads and processes the PubMed abstracts XML baseline from the NCBI FTP server.
# It extracts detailed, structured information for each article.
# It is fully RESUMABLE and uses a robust download-then-process strategy to prevent network errors.

import os
import json
import time
import requests
import gzip
import xml.etree.ElementTree as ET
from collections import defaultdict

# --- Configuration ---
# This points to the PubMed baseline abstracts, not the PMC full-text.
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
# This list generates URLs for the entire PubMed 2024 baseline (~1219 files).
# You can shorten this list for a smaller test run, e.g., range(1, 11) for the first 10 files.
FILE_URLS = [f"{BASE_URL}pubmed25n{i:04d}.xml.gz" for i in range(1, 1220)]


def download_file_with_retries(url, temp_path, max_retries=10, base_delay=10):
    """Downloads a single file with a robust retry mechanism."""
    for attempt in range(max_retries):
        try:
            print(f"  Attempting to download {url.split('/')[-1]}...")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(temp_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"  Download successful.")
            return True
        except (requests.exceptions.RequestException, ConnectionError) as e:
            if attempt + 1 == max_retries:
                print(f"  Download failed after {max_retries} attempts: {e}")
                return False
            delay = base_delay * (2 ** attempt)
            print(f"  Download failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    return False

def extract_detailed_info_from_xml(element):
    """
    Extracts structured information from a <PubmedArticle> XML element.
    """
    # Helper to safely get text, handling missing elements and nested tags
    def get_text(xpath):
        node = element.find(xpath)
        return "".join(node.itertext()).strip() if node is not None else None

    # Extract Authors and their affiliations
    authors = []
    affiliations = set()
    author_list = element.findall('.//Author')
    for author in author_list:
        lastname = author.find('LastName')
        forename = author.find('ForeName')
        if lastname is not None and forename is not None:
            authors.append(f"{lastname.text}, {forename.text}")
        
        affiliation_info = author.find('.//AffiliationInfo/Affiliation')
        if affiliation_info is not None:
            affiliations.add(affiliation_info.text.strip())

    # Extract Publication Date
    pub_date = {}
    pub_date_node = element.find('.//PubDate')
    if pub_date_node is not None:
        year = pub_date_node.find('Year')
        month = pub_date_node.find('Month')
        day = pub_date_node.find('Day')
        if year is not None: pub_date['year'] = year.text
        if month is not None: pub_date['month'] = month.text
        if day is not None: pub_date['day'] = day.text

    # Extract Abstract Text
    abstract_parts = [
        "".join(p.itertext()).strip() for p in element.findall(".//AbstractText")
    ]
    abstract = " ".join(filter(None, abstract_parts))

    # Compile all data into a dictionary
    record = {
        'pmid': get_text('.//PMID'),
        'title': get_text('.//ArticleTitle'),
        'abstract': abstract or None,
        'authors': authors or None,
        'journal': get_text('.//Journal/Title'),
        'publication_date': pub_date or None,
        'affiliations': list(affiliations) or None
    }
    return record

def main():
    """
    Downloads, processes, and saves PubMed abstracts with structured data.
    """
    output_file = "pubmed_abstract_corpus_detailed.jsonl"
    temp_file = "temp_pubmed_file.xml.gz"

    # --- Resumption Logic ---
    # Determines which file and record to start from
    resume_from_count = 0
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for _ in f: resume_from_count += 1
        print(f"Found existing file '{output_file}' with {resume_from_count} articles.")
    except FileNotFoundError:
        print(f"No existing file found. Starting from scratch.")

    records_seen = 0
    newly_processed_count = 0
    
    with open(output_file, 'a', encoding='utf-8') as f:
        for file_index, url in enumerate(FILE_URLS):
            # Fast approximation to skip entire files that have already been processed
            if records_seen + 30000 < resume_from_count:
                records_seen += 30000
                print(f"Skipping file {file_index + 1}/{len(FILE_URLS)} (already processed).")
                continue

            print(f"\nProcessing file {file_index + 1}/{len(FILE_URLS)}: {url.split('/')[-1]}")
            
            # 1. Download the file completely with retries
            if not download_file_with_retries(url, temp_file):
                print(f"Could not download {url}. Skipping to next file.")
                continue

            # 2. Process the downloaded local file
            try:
                with gzip.open(temp_file, 'rb') as local_f:
                    context = ET.iterparse(local_f, events=("end",))
                    for _, elem in context:
                        if elem.tag == 'PubmedArticle':
                            records_seen += 1
                            if records_seen <= resume_from_count:
                                continue

                            record = extract_detailed_info_from_xml(elem)
                            
                            # Only write if essential information is present
                            if record and record.get('title') and record.get('abstract'):
                                f.write(json.dumps(record) + '\n')
                                newly_processed_count += 1

                            elem.clear() # Keep memory usage low

                            if newly_processed_count > 0 and newly_processed_count % 1000 == 0:
                                total_in_file = resume_from_count + newly_processed_count
                                print(f"  ... saved {newly_processed_count} new articles this session (Total: {total_in_file})")
            
            except Exception as e:
                print(f"  An error occurred while parsing {temp_file}: {e}")
            
            finally:
                # 3. Clean up the temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    print("\n-------------------------------------")
    print("Processing complete.")
    total_articles = resume_from_count + newly_processed_count
    print(f"The corpus file '{output_file}' now contains a total of {total_articles} structured articles.")
    print("-------------------------------------\n")

if __name__ == "__main__":
    main()

