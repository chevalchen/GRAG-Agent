import os

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.legacy.agent.recipe_ai_agent import CookingStep, IngredientInfo, RecipeInfo, RecipeKnowledgeGraphBuilder


def create_builder(api_key: str, base_url: str, output_dir: str, batch_size: int) -> RecipeKnowledgeGraphBuilder:
    ai_agent = _ChatRecipeAgent(api_key, base_url)
    return RecipeKnowledgeGraphBuilder(ai_agent, output_dir, batch_size)


class _RecipeParseInput(BaseModel):
    file_path: str = Field(..., description="菜谱 markdown 文件绝对路径")


class _ChatRecipeAgent:
    def __init__(self, api_key: str, base_url: str):
        from langchain_openai import ChatOpenAI

        self.excluded_directories = ["template", ".github", "tips", "starsystem"]
        self._llm = ChatOpenAI(
            model="kimi-k2-0711-preview",
            api_key=api_key,
            base_url=base_url,
            temperature=0.2,
            max_tokens=2048,
        )

    def extract_recipe_info(self, markdown_content: str, file_path: str = "") -> RecipeInfo:
        prompt = "\n".join(
            [
                "你是一个菜谱结构化抽取器。只输出严格 JSON，不要输出其他内容。",
                'JSON schema: {"name": str, "difficulty": int(1-5), "category": str, "cuisine_type": str, "prep_time": str, "cook_time": str, "servings": str, "tags": list[str], "ingredients": list[{"name": str, "amount": str, "unit": str, "category": str, "is_main": bool}], "steps": list[{"step_number": int, "name": str, "description": str, "methods": list[str], "tools": list[str], "time_estimate": str}]}',
                "要求：",
                "- difficulty 若无法判断，返回 3",
                "- ingredients/steps 若缺失，返回空数组",
                "",
                f"文件路径：{file_path}",
                "菜谱内容：",
                markdown_content,
            ]
        )
        rsp = self._llm.invoke(prompt)
        text = getattr(rsp, "content", "") or ""
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < 0 or end <= start:
            payload = {}
        else:
            import json

            try:
                payload = json.loads(text[start : end + 1])
            except Exception:
                payload = {}

        name = str(payload.get("name") or "未知菜谱")
        try:
            difficulty = int(payload.get("difficulty") or 3)
        except Exception:
            difficulty = 3
        difficulty = min(max(difficulty, 1), 5)
        category = str(payload.get("category") or "其他")
        cuisine_type = str(payload.get("cuisine_type") or "")
        prep_time = str(payload.get("prep_time") or "")
        cook_time = str(payload.get("cook_time") or "")
        servings = str(payload.get("servings") or "")
        tags = payload.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        tags = [str(x).strip() for x in tags if str(x).strip()]

        ingredients_in = payload.get("ingredients") or []
        if not isinstance(ingredients_in, list):
            ingredients_in = []
        ingredients: list[IngredientInfo] = []
        for ing in ingredients_in:
            if not isinstance(ing, dict):
                continue
            ingredients.append(
                IngredientInfo(
                    name=str(ing.get("name") or ""),
                    amount=str(ing.get("amount") or ""),
                    unit=str(ing.get("unit") or ""),
                    category=str(ing.get("category") or ""),
                    is_main=bool(ing.get("is_main") or False),
                )
            )

        steps_in = payload.get("steps") or []
        if not isinstance(steps_in, list):
            steps_in = []
        steps: list[CookingStep] = []
        for s in steps_in:
            if not isinstance(s, dict):
                continue
            try:
                step_number = int(s.get("step_number") or 0)
            except Exception:
                step_number = 0
            methods = s.get("methods") or []
            tools = s.get("tools") or []
            if not isinstance(methods, list):
                methods = []
            if not isinstance(tools, list):
                tools = []
            steps.append(
                CookingStep(
                    step_number=step_number,
                    name=str(s.get("name") or f"步骤{step_number}"),
                    description=str(s.get("description") or ""),
                    methods=[str(x).strip() for x in methods if str(x).strip()],
                    tools=[str(x).strip() for x in tools if str(x).strip()],
                    time_estimate=str(s.get("time_estimate") or ""),
                )
            )

        return RecipeInfo(
            name=name,
            difficulty=difficulty,
            category=category,
            cuisine_type=cuisine_type,
            prep_time=prep_time,
            cook_time=cook_time,
            servings=servings,
            ingredients=ingredients,
            steps=steps,
            tags=tags,
        )


class RecipeParseTool(BaseTool):
    name: str = "recipe_parse"
    description: str = "解析单个菜谱 Markdown 为结构化 RecipeInfo"
    args_schema: type[BaseModel] = _RecipeParseInput

    def __init__(self, api_key: str, base_url: str, recipe_root: str):
        self._recipe_root = recipe_root
        self._agent = _ChatRecipeAgent(api_key, base_url)
        super().__init__()

    def parse(self, file_path: str) -> dict:
        return self.invoke({"file_path": file_path})

    def _run(self, file_path: str) -> dict:
        rel_path = os.path.relpath(file_path, self._recipe_root)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            recipe_info = self._agent.extract_recipe_info(content, rel_path)
            return {"ok": True, "rel_path": rel_path, "recipe_info": recipe_info, "error": None}
        except Exception as e:
            return {"ok": False, "rel_path": rel_path, "recipe_info": None, "error": str(e)}
