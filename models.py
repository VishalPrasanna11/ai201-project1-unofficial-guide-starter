from dataclasses import dataclass


@dataclass
class Document:
    filename: str
    raw_text: str
    source_path: str


@dataclass
class Chunk:
    text: str
    source: str
    section: str | None
    chunk_index: int

    def with_metadata_prefix(self) -> "Chunk":
        section_part = f" | Section: {self.section}" if self.section else ""
        prefix = f"[Source: {self.source}{section_part}]\n"
        if self.text.startswith(prefix):
            return self
        return Chunk(
            text=prefix + self.text,
            source=self.source,
            section=self.section,
            chunk_index=self.chunk_index,
        )
