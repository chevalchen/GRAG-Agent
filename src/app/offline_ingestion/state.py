from typing import Annotated, Any, Dict, List, Optional, TypedDict


def _merge_metrics(left: Dict[str, Any] | None, right: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def _pick_error(left: Optional[str], right: Optional[str]) -> Optional[str]:
    return right or left


class OfflineIngestionState(TypedDict, total=False):
    """
    离线索引状态
    
    Attributes:
        recipe_dir: 索引配方目录
        output_dir: 输出目录
        output_format: 输出格式
        resume: 是否恢复
        total_files: 总文件数
        file_list: 文件列表
        next_index: 下一个索引
        processed: 已处理文件数
        failed: 失败文件数
        current_batch_count: 当前批次文件数
        indexed_count: 已索引文件数
        failed_writes: 失败写入列表
        metrics: 指标
        error: 错误信息
    """
    recipe_dir: str
    output_dir: str
    output_format: str
    resume: bool
    total_files: int
    file_list: List[str]
    medical_entities_path: str
    literature_path: str
    health_chunk_records: List[Dict[str, Any]]
    next_index: int
    processed: int
    failed: int
    current_batch_count: int
    indexed_count: int
    health_indexed_count: int
    failed_writes: List[Dict[str, Any]]
    parsed_records: List[Dict[str, Any]]
    chunk_records: List[Dict[str, Any]]
    metrics: Annotated[Dict[str, Any], _merge_metrics]
    error: Annotated[Optional[str], _pick_error]
