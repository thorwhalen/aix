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


def verify_urls(md_string):
    """
    Verifies URLs in a markdown string by checking their status codes.

    Args:
        md_string (str): The markdown string containing URLs.

    Returns:
        dict: A dictionary with URLs as keys and their status codes as values.
    """
    # Regex to extract URLs from the markdown string
    pattern = r"\[(.*?)\]\((.*?)\)"
    matches = re.findall(pattern, md_string)

    url_status_codes = {}

    for title, url in matches:
        try:
            response = requests.head(url, allow_redirects=True)
            url_status_codes[url] = response.status_code
        except Exception as e:
            url_status_codes[url] = str(e)

    return url_status_codes


# --------------------------------------------------------------------------------------
# Code

from typing import Mapping
import os
from typing import Any, Union, Callable
from dol import store_aggregate, TextFiles, filt_iter, cached_keys

DirectoryPathString = str
GithubUrl = str
RegexString = str
CodeSource = Union[DirectoryPathString, GithubUrl, Mapping]


def code_aggregate(md_string):
    return store_aggregate(md_string.splitlines())


def resolve_code_source_dir_path(code_src: CodeSource) -> DirectoryPathString:
    """
    Resolves code_src to a directory path string.

    I
    Args:
        code_src (Any): The source of the code. Can be
            a directory path string,
            a GitHub URL,
            or an imported package (must contain a __path__ atribute)

    Returns:
        The resolved directory path.

    Raises:
        AssertionError: If the resolved path is not a valid directory.
    """
    if isinstance(code_src, str):
        # If it's a string, check if it's a directory
        if os.path.isdir(code_src):
            return os.path.abspath(code_src)  # Return absolute path if it's a directory
        elif '\n' not in code_src and 'github' in code_src:
            from hubcap import ensure_repo_folder  # pip install hubcap

            # If it's a GitHub URL, download the repository
            repo_url = code_src
            repo_path = ensure_repo_folder(repo_url)
            return repo_path
        else:
            raise ValueError(f"Unsupported string format or non-directory: {code_src}")
    # If it's not a string, check for __path__ attribute
    elif hasattr(code_src, "__path__"):
        path_list = list(code_src.__path__)
        assert len(path_list) == 1, (
            f"The __path__ attribute should contain exactly one path, "
            f"but found: {path_list}"
        )
        return os.path.abspath(path_list[0])

    # If no valid resolution was found
    raise ValueError(
        f"Unable to resolve code_src to a valid directory path: {code_src}"
    )


def resolve_code_source(
    code_src: CodeSource, keys_filt: Callable = lambda x: x.endswith('.py')
) -> Mapping:
    """
    Will resolve code_src to a Mapping whose values are the code strings

    Args:
        code_src: The source of the code. Can be an explicit `Mapping`,
            a directory path string,
            a GitHub URL,
            or an imported package (must contain a __path__ atribute)
        keys_filt (Callable): A function to filter the keys. Defaults to lambda x: x.endswith('.py').

    """
    if isinstance(code_src, Mapping):
        return code_src
    else:
        code_src_rootdir = resolve_code_source_dir_path(code_src)
        return cached_keys(
            filt_iter(TextFiles(code_src_rootdir), filt=keys_filt), keys_cache=sorted
        )


def identity(x):
    return x


def aggregate_code(
    code_src: CodeSource,
    *,
    egress: Union[Callable, str] = lambda x: x,
    kv_to_item=lambda k, v: f"## {k}\n\n```python\n{v.strip()}\n```",
    **store_aggregate_kwargs,
) -> Any:
    """
    Aggregates all code segments from the given code source (folder, github url, store).

    Args:
        code_src (dict): A dictionary where keys are references to the code (e.g., paths)
                         and values are code snippets or content.
        egress (Union[Callable, str]): A function to apply to the aggregate before returning.
                                       If a string, the aggregate will be saved to the file.
        kv_to_item (Callable): A function that converts a key-value pairs to the
                               items that should be aggregated.
        **store_aggregate_kwargs: Additional keyword arguments to pass to store_aggregate.

    See dol.store_aggregate for more details.

    Returns:
        Any: The aggregated code content, or the result of the egress function.

    Example:

    >>> code_src = {
    ...     'module1.py': 'def foo(): pass',
    ...     'module2.py': 'def bar(): pass',
    ...     'module3.py': 'class Baz: pass',
    ... }
    >>> print(aggregate_code(code_src))
    ## module1.py
    <BLANKLINE>
    ```python
    def foo(): pass
    ```
    <BLANKLINE>
    ## module2.py
    <BLANKLINE>
    ```python
    def bar(): pass
    ```
    <BLANKLINE>
    ## module3.py
    <BLANKLINE>
    ```python
    class Baz: pass
    ```

    Here, let's input an imported (third party) package, and have the function save the
    result to a temporary file

    >>> from tempfile import NamedTemporaryFile
    >>> temp_file_name = NamedTemporaryFile().name
    >>> import aix
    >>> _  = aggregate_code(aix, egress=temp_file_name)
    >>> print(open(temp_file_name).read(25))
    ## __init__.py
    <BLANKLINE>
    ```python

    If you have hubcap installed, you can even get an aggregate of code from a GitHub
    repository.

    >>> string = aggregate_code('https://github.com/thorwhalen/aix')  # doctest: +SKIP


    """
    code_store = resolve_code_source(code_src)
    return store_aggregate(
        code_store, egress=egress, kv_to_item=kv_to_item, **store_aggregate_kwargs
    )
