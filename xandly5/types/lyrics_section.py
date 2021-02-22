
from typing import Optional
from enum import IntEnum
import uuid


# TODO AEO move to separate file
class LyricsSectionType(IntEnum):
    VERSE = 1
    CHORUS = 2
    BRIDGE = 3
    INTRO = 4
    PRE_CHORUS = 5
    OUTRO = 6


class LyricsSection:

    def __init__(self, section_type: LyricsSectionType, word_count: int, word_group_count: int, seed_text: str):
        self.section_id = str(uuid.uuid4())
        self.section_type = section_type
        self.word_count = word_count
        self.word_group_count = word_group_count
        self.seed_text = seed_text
        self.generated_text: Optional[str] = None
