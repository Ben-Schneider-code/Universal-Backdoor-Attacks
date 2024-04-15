import re


def is_valid_url(string):
    url_regex = re.compile(r'^(http|https)://[a-zA-Z0-9._-]+\.[a-zA-Z]{2,5}$')
    return url_regex.match(string)