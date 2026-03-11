from typing import Any, Dict, List, Optional, TypedDict


class OfflineIngestionState(TypedDict):
    recipe_dir: str
    output_dir: str
    output_format: str
    resume: bool
    total_files: int
    file_list: List[str]
    next_index: int
    processed: int
    failed: int
    current_batch_count: int
    metrics: Dict[str, Any]
    error: Optional[str]
