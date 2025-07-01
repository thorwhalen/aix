"""
Tools to make contexts (knowledge bases for AI agents)

Examples:

Download all articles from a markdown string and save them as PDF files:

>>> download_articles(md_string)  # doctest: +SKIP

Verify URLs in a markdown string by checking their status codes
(useful when trying to verify if AI hallucinated the urls)
>>> verify_urls(md_string)  # doctest: +SKIP

Make an md file with all the code in a directory:

>>> md_string = code_aggregate(package_or_folder_or_github_url)  # doctest: +SKIP

"""

from contaix import *