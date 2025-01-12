"""Tools to make contexts (knowledge bases for AI agents)"""

import os
import re
import requests

DFLT_SAVE_DIR = os.path.expanduser('~/Downloads')


def download_articles(
    md_string: str,
    save_dir: str = DFLT_SAVE_DIR,
    *,
    save_non_pdf: bool = False,
    verbose: bool = True,
):
    """
    Downloads articles from the given markdown string and saves them as PDF files.

    Args:
        md_string (str): The markdown-style string containing titles and URLs.
        save_dir (str): The root directory to save the downloaded PDFs. Defaults to '~/Downloads'.
        save_non_pdf (bool): Whether to save non-PDF content. Defaults to False.
        verbose (bool): Whether to print detailed messages. Defaults to True.

    Returns:
        list: A list of URLs that failed to download or were invalid PDFs.

    Example:

    >>> md_string = '''
    ... - **[Valid PDF](https://example.com/file.pdf)**: A valid PDF file.
    ... - **[Invalid PDF](https://example.com/file.html)**: An HTML page, not a PDF.
    ... '''  # doctest: +SKIP
    >>> download_articles(md_string, save_non_pdf=True)  # doctest: +SKIP
    Downloaded: Valid PDF -> ~/Downloads/Valid_PDF.pdf
    Skipped (HTML or non-PDF): Invalid PDF from https://example.com/file.html
    Non-PDF content saved to: ~/Downloads/Invalid_PDF_non_pdf.html

    """
    # Assert the save_dir exists
    assert os.path.exists(save_dir), f"Directory not found: {save_dir}"

    def clog(msg):
        if verbose:
            print(msg)

    # Regex to extract titles and URLs from the markdown string
    pattern = r"- \*\*\[(.*?)\]\((.*?)\)\*\*"
    matches = re.findall(pattern, md_string)

    failed_urls = []

    for title, url in matches:
        # Sanitize title to create a valid filename
        sanitized_title = re.sub(r'[^\w\-_\. ]', '_', title)
        filename = f"{sanitized_title}.pdf"
        filepath = os.path.join(save_dir, filename)

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Check Content-Type header
            content_type = response.headers.get('Content-Type', '')
            if 'application/pdf' not in content_type:
                clog(
                    f"Skipped (HTML or non-PDF): {title} from {url} (Content-Type: {content_type})"
                )
                if save_non_pdf:
                    # Save non-PDF content with a different extension
                    non_pdf_path = os.path.join(
                        save_dir, f"{sanitized_title}_non_pdf.html"
                    )
                    with open(non_pdf_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    clog(f"Non-PDF content saved to: {non_pdf_path}")
                failed_urls.append(url)
                continue

            # Verify PDF content by checking the first few bytes
            first_chunk = next(response.iter_content(chunk_size=8192))
            if not first_chunk.startswith(b'%PDF'):
                clog(f"Invalid PDF content: {title} from {url}")
                if save_non_pdf:
                    # Save invalid PDF content with a different extension
                    invalid_pdf_path = os.path.join(
                        save_dir, f"{sanitized_title}_invalid.pdf"
                    )
                    with open(invalid_pdf_path, 'wb') as f:
                        f.write(first_chunk)
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    clog(f"Invalid PDF content saved to: {invalid_pdf_path}")
                failed_urls.append(url)
                continue

            # Save the content as a PDF file
            with open(filepath, 'wb') as f:
                f.write(first_chunk)  # Write the first chunk already read
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            clog(f"Downloaded: {title} -> {filepath}")
        except Exception as e:
            clog(f"Failed to download {title} from {url}: {e}")
            failed_urls.append(url)

    return failed_urls


def download_articles_by_section(
    md_string, rootdir=None, save_non_pdf: bool = False, *, section_marker: str = r"###"
):
    """
    Downloads articles from a markdown string organized by sections into subdirectories.

    Args:
        md_string (str): The markdown string with sections and articles.
        rootdir (str): The root directory where subdirectories for sections will be created.
                       Defaults to '~/Downloads'.
        save_non_pdf (bool): Whether to save non-PDF content. Defaults to False.

    Returns:
        dict: A dictionary with section names as keys and lists of failed URLs as values.
    """
    if rootdir is None:
        rootdir = os.path.expanduser('~/Downloads')

    # Ensure the root directory exists
    os.makedirs(rootdir, exist_ok=True)

    # Parse sections and their content
    section_pattern = section_marker + r" (.*?)\n(.*?)(?=\n" + section_marker + "|\Z)"
    sections = re.findall(section_pattern, md_string, re.DOTALL)

    failed_urls_by_section = {}

    for section_title, section_content in sections:
        # Create a snake-case directory name for the section
        sanitized_section_title = (
            re.sub(r'[^\w\s]', '', section_title).strip().replace(' ', '_').lower()
        )
        section_dir = os.path.join(rootdir, sanitized_section_title)
        os.makedirs(section_dir, exist_ok=True)

        print(f"\nProcessing section: {section_title} (Directory: {section_dir})")

        # Download articles for this section
        failed_urls = download_articles(
            section_content, save_dir=section_dir, save_non_pdf=save_non_pdf
        )
        failed_urls_by_section[section_title] = failed_urls

    return failed_urls_by_section
