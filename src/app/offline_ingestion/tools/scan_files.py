import os
from typing import List, Sequence, Tuple


def scan_recipe_files(recipe_dir: str, excluded_directories: Sequence[str]) -> List[Tuple[str, str]]:
    dishes_dir = os.path.join(recipe_dir, "dishes")
    scan_root = dishes_dir if os.path.exists(dishes_dir) else recipe_dir

    results: List[Tuple[str, str]] = []
    for root, dirs, files in os.walk(scan_root):
        rel_to_recipe = os.path.relpath(root, recipe_dir)
        path_parts = rel_to_recipe.replace("\\", "/").split("/")
        if any(excluded in path_parts for excluded in excluded_directories):
            dirs[:] = []
            continue

        for name in files:
            if not name.lower().endswith(".md"):
                continue
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, recipe_dir)
            rel_parts = rel_path.replace("\\", "/").split("/")
            if any(excluded in rel_parts for excluded in excluded_directories):
                continue
            results.append((abs_path, rel_path))

    results.sort(key=lambda x: x[1])
    return results

