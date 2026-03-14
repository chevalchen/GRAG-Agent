from typing import Any, Dict, List, Optional, TypedDict


class OfflineIngestionState(TypedDict, total=False):
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
    indexed_count: int
    failed_writes: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    error: Optional[str]
