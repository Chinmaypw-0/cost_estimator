import os
from typing import Literal
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

class Task(BaseModel):
    name: str
    role: Literal["developer", "tester", "project_manager", "designer", "devops"]
    complexity: Literal["low", "medium", "high"]

class Breakdown(BaseModel):
    domain: Literal["business_software", "web_mobile", "data_ai",
                    "embedded_iot", "engineering_scientific", "other"]
    domain_confidence: float = Field(ge=0, le=1)
    size_unit: Literal["fp", "kloc"]
    size_value: float = Field(gt=0)
    size_reasoning: str
    suggested_team_size: int = Field(ge=1, le=20)
    tasks: list[Task]

PROMPT = """You help estimate software project cost. Read the project description and
return ONLY a JSON object with these keys:
domain: one of business_software, web_mobile, data_ai, embedded_iot, engineering_scientific, other
domain_confidence: number from 0 to 1 (use a low value if the description is vague)
size_unit: "fp" (function points) for business_software, web_mobile, data_ai, other;
           "kloc" (thousand lines of code) for embedded_iot and engineering_scientific
size_value: your size estimate in that unit
size_reasoning: one short sentence on how you got the size
suggested_team_size: whole number of people
tasks: list of {name, role, complexity}, role one of developer, tester,
       project_manager, designer, devops; complexity one of low, medium, high

Reference sizes: a small CRUD web app is about 100-200 function points, a medium
business system about 300-600, a large one 1000 or more. A small embedded firmware
project is about 5-20 KLOC, a large one 100 KLOC or more. Be realistic, not generous."""

def decompose(description, tries=2):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    last = None
    for _ in range(tries):
        r = client.models.generate_content(
            model=model,
            contents=PROMPT + "\n\nProject description:\n" + description,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0),
        )
        try:
            return Breakdown.model_validate_json(r.text)
        except Exception as e:      # bad or incomplete JSON: try once more
            last = e
    raise last

if __name__ == "__main__":
    text = ("An inventory management web app for a small shop with login, product "
            "add/edit/delete, stock alerts, sales reports and a mobile-friendly design.")
    print(decompose(text).model_dump_json(indent=2))