# models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source_name: str

    page: Optional[int] = None
    section: Optional[str] = None
    heading_path: Optional[str] = None

    block_type: str = "paragraph"
    order: Optional[int] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
