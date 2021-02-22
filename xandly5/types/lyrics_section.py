
from typing import Optional
import uuid
from xandly5.types.section_type_enum import SectionTypeEnum


class LyricsSection:

    def __init__(self, section_type: SectionTypeEnum, word_count: int, word_group_count: int, seed_text: str):
        self.section_id = str(uuid.uuid4())
        self.section_type = section_type
        self.word_count = word_count
        self.word_group_count = word_group_count
        self.seed_text = seed_text
        self.generated_text: Optional[str] = None
