from dataclasses import dataclass


@dataclass
class RegexPatterns:

    URL = r'https?://(?:www\.)?[a-zA-Z0-9./]+'
