import os

from src.legacy.agent.recipe_ai_agent import KimiRecipeAgent, RecipeKnowledgeGraphBuilder


def create_builder(api_key: str, base_url: str, output_dir: str, batch_size: int) -> RecipeKnowledgeGraphBuilder:
    ai_agent = KimiRecipeAgent(api_key, base_url)
    return RecipeKnowledgeGraphBuilder(ai_agent, output_dir, batch_size)


class RecipeParseTool:
    def __init__(self, api_key: str, base_url: str, recipe_root: str):
        self._recipe_root = recipe_root
        self._agent = KimiRecipeAgent(api_key, base_url)

    def parse(self, file_path: str) -> dict:
        rel_path = os.path.relpath(file_path, self._recipe_root)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            recipe_info = self._agent.extract_recipe_info(content, rel_path)
            return {"ok": True, "rel_path": rel_path, "recipe_info": recipe_info, "error": None}
        except Exception as e:
            return {"ok": False, "rel_path": rel_path, "recipe_info": None, "error": str(e)}
