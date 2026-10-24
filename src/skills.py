"""Skill vocabulary and skill detection.

Skills are stored as  {category: {display name: [aliases]}}.
Detection is regex based with strict boundaries so that "java" does not match
"javascript", "sql" does not match "mysql", and "C++" / "C#" / ".NET" work.

To extend the system, just add entries to SKILL_DB.
"""
from __future__ import annotations

import re

SOFT_CATEGORY = "Soft Skills"

SKILL_DB: dict[str, dict[str, list[str]]] = {
    "Programming Languages": {
        "Python": [], "Java": [], "JavaScript": ["js", "ecmascript"], "TypeScript": ["ts"],
        "C++": ["cpp"], "C#": ["csharp"], "Golang": [], "Rust": [], "Kotlin": [], "Swift": [],
        "PHP": [], "Ruby": [], "Scala": [], "MATLAB": [], "R Programming": ["r language"],
        "Bash": ["shell scripting", "shell script"], "SQL": [],
    },
    "Web & Backend": {
        "HTML": ["html5"], "CSS": ["css3"], "React": ["react.js", "reactjs"], "Angular": ["angularjs"],
        "Vue.js": ["vue", "vuejs"], "Node.js": ["nodejs", "node js"], "Express.js": ["expressjs"],
        "Django": [], "Flask": [], "FastAPI": [], "Spring Boot": ["springboot"],
        ".NET": ["dotnet", "asp.net"], "REST APIs": ["rest api", "restful api", "restful apis", "restful services"],
        "GraphQL": [], "Microservices": ["microservice"],
    },
    "Databases & Big Data": {
        "MySQL": [], "PostgreSQL": ["postgres"], "MongoDB": ["mongo"], "Redis": [], "SQLite": [],
        "Cassandra": [], "Elasticsearch": [], "Snowflake": [], "NoSQL": [], "Hadoop": [],
        "Spark": ["apache spark", "pyspark"], "Kafka": ["apache kafka"], "Airflow": ["apache airflow"],
        "ETL": [], "Data Warehousing": ["data warehouse"], "Data Modeling": ["data modelling"],
    },
    "Data Science & AI": {
        "Machine Learning": ["ml"], "Deep Learning": [], "Natural Language Processing": ["nlp"],
        "Computer Vision": [], "Generative AI": ["genai", "gen ai"],
        "LLMs": ["llm", "large language model", "large language models"],
        "Reinforcement Learning": [], "Time Series": ["time series analysis", "time-series"],
        "Statistics": ["statistical analysis", "statistical modeling"], "Data Analysis": ["data analytics"],
        "Data Visualization": ["data visualisation"], "Feature Engineering": [], "A/B Testing": ["ab testing"],
        "Text Classification": [], "Pandas": [], "NumPy": [], "SciPy": [],
        "scikit-learn": ["sklearn", "scikit learn"], "TensorFlow": [], "PyTorch": ["torch"], "Keras": [],
        "XGBoost": [], "spaCy": [], "NLTK": [], "Hugging Face": ["huggingface"], "OpenCV": [],
        "LangChain": [], "RAG": ["retrieval augmented generation", "retrieval-augmented generation"],
        "Matplotlib": [], "Seaborn": [], "Plotly": [], "Streamlit": [],
    },
    "BI & Analytics Tools": {
        "Tableau": [], "Power BI": ["powerbi"], "Excel": ["ms excel", "microsoft excel", "advanced excel"],
        "Looker": [], "Google Analytics": [],
    },
    "Cloud & DevOps": {
        "AWS": ["amazon web services"], "Azure": ["microsoft azure"], "Google Cloud": ["gcp", "google cloud platform"],
        "Docker": [], "Kubernetes": ["k8s"], "Terraform": [], "Jenkins": [], "CI/CD": ["cicd", "ci cd"],
        "Git": [], "GitHub": [], "GitLab": [], "Linux": [], "Ansible": [], "MLOps": [], "DevOps": [],
    },
    "Testing & Process": {
        "Selenium": [], "JUnit": [], "Pytest": [], "Postman": [], "Jira": [], "Agile": ["scrum"],
        "Unit Testing": [], "Cybersecurity": ["cyber security", "information security"],
    },
    SOFT_CATEGORY: {
        "Communication": ["communication skills"], "Leadership": [], "Teamwork": ["team player"],
        "Problem Solving": ["problem-solving"], "Project Management": [], "Stakeholder Management": [],
        "Time Management": [],
    },
}

# Names that are also normal English words: match only as "Name" / "NAME" (case-sensitive).
AMBIGUOUS = {"React", "Excel", "Spark", "Swift", "Rust", "Ruby", "Flask", "Looker"}

_L = r"(?<![A-Za-z0-9+#])"
_R = r"(?![A-Za-z0-9+#])"


def _alt(terms: list[str]) -> str:
    terms = sorted(set(terms), key=len, reverse=True)
    return "|".join(re.escape(t).replace(r"\ ", r"[\s-]+") for t in terms)


class _Skill:
    __slots__ = ("name", "category", "ci", "cs")

    def __init__(self, name: str, category: str, aliases: list[str]):
        self.name, self.category = name, category
        if name in AMBIGUOUS:
            self.cs = re.compile(_L + "(?:" + _alt([name, name.upper()]) + ")" + _R)
            ci_terms = aliases
        else:
            self.cs = None
            ci_terms = [name] + aliases
        self.ci = re.compile(_L + "(?:" + _alt(ci_terms) + ")" + _R, re.I) if ci_terms else None

    def found_in(self, text: str) -> bool:
        return bool((self.ci and self.ci.search(text)) or (self.cs and self.cs.search(text)))


_SKILLS = [_Skill(n, cat, al) for cat, group in SKILL_DB.items() for n, al in group.items()]


def find_skills(text: str, include_soft: bool = False) -> list[str]:
    """Return the sorted list of canonical skill names found in ``text``."""
    found = [s.name for s in _SKILLS if (include_soft or s.category != SOFT_CATEGORY) and s.found_in(text)]
    return sorted(found, key=str.lower)
