"""Skill taxonomy: canonical names, aliases, categories, families and implications.

The taxonomy is the single source of truth for skill detection and normalisation. It is
deliberately small and transparent so every match can be traced to an alias.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# (canonical, category, aliases...)  -- aliases are matched case-insensitively
_RAW = [
    # Programming languages
    ("python", "Programming", "python3", "py"),
    ("java", "Programming", "core java", "java se", "j2ee"),
    ("javascript", "Programming", "js", "es6", "ecmascript"),
    ("typescript", "Programming", "ts"),
    ("c++", "Programming", "cpp"),
    ("c#", "Programming", "csharp", "c sharp"),
    ("go", "Programming", "golang"),
    ("rust", "Programming"),
    ("kotlin", "Programming"),
    ("swift", "Programming"),
    ("scala", "Programming"),
    ("php", "Programming"),
    ("ruby", "Programming"),
    ("r", "Programming", "r programming", "r language"),
    ("matlab", "Programming"),
    ("bash", "Programming", "shell scripting", "shell script", "shell"),
    ("c", "Programming"),
    # ML / AI
    ("machine learning", "ML/AI", "ml", "applied machine learning", "statistical learning", "predictive modeling", "predictive modelling"),
    ("deep learning", "ML/AI", "neural networks", "neural network", "cnn", "rnn", "lstm"),
    ("natural language processing", "ML/AI", "nlp", "text mining", "text analytics"),
    ("computer vision", "ML/AI", "image processing", "image recognition", "object detection", "opencv"),
    ("generative ai", "ML/AI", "genai", "gen ai", "generative models"),
    ("large language models", "ML/AI", "llm", "llms", "gpt", "language models", "foundation models"),
    ("prompt engineering", "ML/AI", "prompt design"),
    ("rag", "ML/AI", "retrieval augmented generation", "retrieval-augmented generation"),
    ("transformers", "ML/AI", "bert", "transformer models", "attention models"),
    ("hugging face", "ML/AI", "huggingface", "hf transformers"),
    ("langchain", "ML/AI", "llamaindex", "llama index"),
    ("pytorch", "ML/AI", "torch"),
    ("tensorflow", "ML/AI", "tf"),
    ("keras", "ML/AI"),
    ("scikit-learn", "ML/AI", "sklearn", "scikit learn"),
    ("xgboost", "ML/AI", "lightgbm", "gradient boosting", "catboost"),
    ("reinforcement learning", "ML/AI", "rl"),
    ("recommendation systems", "ML/AI", "recommender systems", "recommendation engine", "recommender system", "collaborative filtering"),
    ("time series", "ML/AI", "time-series", "forecasting", "arima"),
    ("feature engineering", "ML/AI", "feature selection", "feature extraction"),
    ("model evaluation", "ML/AI", "model validation", "cross-validation", "cross validation", "hyperparameter tuning"),
    ("mlops", "ML/AI", "model deployment", "model serving", "mlflow", "kubeflow", "ml pipelines"),
    ("classification", "ML/AI", "classifier", "classifiers"),
    ("regression", "ML/AI", "linear regression", "logistic regression"),
    ("clustering", "ML/AI", "k-means", "kmeans", "unsupervised learning"),
    # Data
    ("pandas", "Data", "dataframes"),
    ("numpy", "Data"),
    ("data analysis", "Data", "data analytics", "exploratory data analysis", "eda", "analyzing data", "analysing data", "data analyst"),
    ("data visualization", "Data", "data visualisation", "dashboards", "dashboarding", "matplotlib", "seaborn", "plotly"),
    ("statistics", "Data", "statistical analysis", "statistical modeling", "hypothesis testing", "a/b testing", "ab testing"),
    ("data preprocessing", "Data", "data cleaning", "data wrangling", "data preparation", "data cleansing"),
    ("etl", "Data", "elt", "data pipelines", "data pipeline", "data integration"),
    ("apache spark", "Data", "spark", "pyspark"),
    ("hadoop", "Data", "hdfs", "mapreduce", "hive"),
    ("kafka", "Data", "apache kafka", "event streaming"),
    ("airflow", "Data", "apache airflow", "dagster", "prefect"),
    ("dbt", "Data"),
    ("data warehousing", "Data", "data warehouse", "snowflake", "bigquery", "redshift"),
    ("excel", "Data", "microsoft excel", "ms excel", "spreadsheets", "advanced excel"),
    ("power bi", "Data", "powerbi", "power-bi"),
    ("tableau", "Data"),
    ("looker", "Data", "qlik", "qlikview", "metabase"),
    # Databases
    ("sql", "Database", "structured query language", "t-sql", "pl/sql", "sql queries"),
    ("postgresql", "Database", "postgres"),
    ("mysql", "Database", "mariadb"),
    ("sql server", "Database", "mssql", "ms sql", "microsoft sql server"),
    ("sqlite", "Database"),
    ("oracle database", "Database", "oracle db"),
    ("mongodb", "Database", "mongo"),
    ("redis", "Database"),
    ("cassandra", "Database", "dynamodb"),
    ("elasticsearch", "Database", "opensearch", "elk stack"),
    ("faiss", "Database", "pinecone", "chromadb", "weaviate", "milvus"),
    ("vector databases", "Database", "vector database", "vector db", "vector store", "vector search", "embeddings"),
    # Web / backend
    ("html", "Web", "html5"),
    ("css", "Web", "css3", "sass", "scss", "tailwind", "tailwind css", "bootstrap"),
    ("react", "Web", "react.js", "reactjs", "react js", "redux"),
    ("angular", "Web", "angularjs"),
    ("vue", "Web", "vue.js", "vuejs", "nuxt"),
    ("next.js", "Web", "nextjs"),
    ("node.js", "Web", "nodejs", "node js", "node"),
    ("express", "Web", "express.js", "expressjs"),
    ("django", "Web", "django rest framework", "drf"),
    ("flask", "Web"),
    ("fastapi", "Web", "fast api"),
    ("spring boot", "Web", "spring", "spring framework", "hibernate"),
    (".net", "Web", "asp.net", "dotnet", ".net core"),
    ("rest api", "Web", "restful api", "restful apis", "rest apis", "restful services", "rest", "api development", "web services"),
    ("graphql", "Web"),
    ("microservices", "Web", "microservice architecture", "service-oriented architecture"),
    ("streamlit", "Web", "gradio"),
    ("system design", "Web", "distributed systems", "scalable systems", "software architecture"),
    # Cloud / DevOps
    ("aws", "Cloud/DevOps", "amazon web services", "ec2", "s3", "lambda", "sagemaker"),
    ("azure", "Cloud/DevOps", "microsoft azure", "azure ml"),
    ("gcp", "Cloud/DevOps", "google cloud", "google cloud platform", "vertex ai"),
    ("docker", "Cloud/DevOps", "containerization", "containers", "dockerfile", "docker compose"),
    ("kubernetes", "Cloud/DevOps", "k8s", "helm", "eks", "gke", "aks"),
    ("terraform", "Cloud/DevOps", "infrastructure as code", "iac", "ansible", "cloudformation", "pulumi"),
    ("ci/cd", "Cloud/DevOps", "cicd", "ci cd", "continuous integration", "continuous delivery", "continuous deployment", "jenkins", "github actions", "gitlab ci", "circleci"),
    ("linux", "Cloud/DevOps", "unix", "ubuntu"),
    ("monitoring", "Cloud/DevOps", "observability", "prometheus", "grafana", "datadog", "new relic"),
    ("networking", "Cloud/DevOps", "tcp/ip", "dns", "load balancing"),
    # Tools / practices
    ("git", "Tools", "version control", "gitlab", "bitbucket"),
    ("github", "Tools"),
    ("jira", "Tools", "confluence"),
    ("data structures", "CS Fundamentals", "data structures and algorithms", "dsa"),
    ("algorithms", "CS Fundamentals", "algorithm design", "competitive programming"),
    ("object-oriented programming", "CS Fundamentals", "oop", "oops", "object oriented programming", "object oriented design", "design patterns"),
    ("operating systems", "CS Fundamentals"),
    ("unit testing", "Testing", "pytest", "junit", "jest", "unittest", "test-driven development", "tdd", "test automation", "mocha"),
    ("selenium", "Testing", "cypress", "playwright", "qa automation"),
    ("agile", "Practices", "scrum", "kanban", "sprint planning"),
    ("project management", "Practices", "program management", "product management", "stakeholder management"),
    ("leadership", "Practices", "team lead", "team leadership", "mentoring", "mentored"),
    ("android", "Mobile", "android development", "android studio"),
    ("ios", "Mobile", "ios development", "swiftui", "xcode"),
    ("react native", "Mobile", "flutter"),
    ("cybersecurity", "Security", "information security", "penetration testing", "owasp", "network security"),
    ("figma", "Design", "adobe xd", "sketch app", "prototyping"),
    ("ux design", "Design", "ui/ux", "ux/ui", "ux design", "ui design", "interaction design", "wireframing", "wireframes"),
    ("user research", "Design", "usability testing", "user interviews", "user testing"),
]

# Groups of closely related skills: a missing requirement with a related skill present
# earns partial credit (RELATED_CREDIT) and is reported as "related evidence".
FAMILIES = {
    "deep learning frameworks": ["pytorch", "tensorflow", "keras", "deep learning"],
    "classical ML": ["machine learning", "scikit-learn", "xgboost", "classification", "regression", "clustering", "model evaluation", "feature engineering", "time series", "statistics"],
    "language models": ["natural language processing", "large language models", "transformers", "hugging face", "langchain", "rag", "generative ai", "prompt engineering"],
    "vector search": ["faiss", "vector databases", "elasticsearch"],
    "cloud platforms": ["aws", "azure", "gcp"],
    "relational databases": ["sql", "postgresql", "mysql", "sql server", "sqlite", "oracle database"],
    "nosql databases": ["mongodb", "redis", "cassandra"],
    "python web frameworks": ["fastapi", "flask", "django"],
    "frontend frameworks": ["react", "angular", "vue", "next.js", "javascript", "typescript"],
    "server-side javascript": ["node.js", "express", "javascript", "typescript"],
    "containers": ["docker", "kubernetes"],
    "infrastructure automation": ["terraform", "ci/cd", "kubernetes", "docker", "linux"],
    "BI tools": ["power bi", "tableau", "looker", "data visualization", "excel"],
    "big data": ["apache spark", "hadoop", "kafka", "airflow", "etl", "dbt", "data warehousing"],
    "JVM": ["java", "kotlin", "scala", "spring boot"],
    "C family": ["c", "c++", "c#"],
    "backend APIs": ["rest api", "graphql", "microservices", "fastapi", "flask", "django", "node.js", "spring boot"],
    "testing": ["unit testing", "selenium"],
    "data analysis": ["data analysis", "pandas", "numpy", "data preprocessing", "statistics", "sql", "excel"],
    "mobile": ["android", "ios", "react native", "kotlin", "swift"],
    "design": ["figma", "ux design", "user research"],
}

# If a resume evidences key, it (strongly) implies each skill in the value list.
IMPLIES = {
    "pytorch": ["deep learning", "machine learning", "python"],
    "tensorflow": ["deep learning", "machine learning"],
    "keras": ["deep learning", "machine learning"],
    "deep learning": ["machine learning"],
    "scikit-learn": ["machine learning", "python"],
    "xgboost": ["machine learning"],
    "pandas": ["python", "data analysis"],
    "numpy": ["python"],
    "hugging face": ["transformers", "natural language processing", "python"],
    "transformers": ["natural language processing", "deep learning"],
    "large language models": ["natural language processing", "generative ai"],
    "langchain": ["large language models", "python"],
    "rag": ["large language models"],
    "django": ["python", "rest api"],
    "flask": ["python"],
    "fastapi": ["python", "rest api"],
    "react": ["javascript", "html", "css"],
    "next.js": ["react", "javascript"],
    "angular": ["javascript", "typescript"],
    "vue": ["javascript"],
    "node.js": ["javascript"],
    "express": ["node.js", "javascript"],
    "typescript": ["javascript"],
    "spring boot": ["java"],
    "postgresql": ["sql"],
    "mysql": ["sql"],
    "sql server": ["sql"],
    "sqlite": ["sql"],
    "oracle database": ["sql"],
    "kubernetes": ["docker"],
    "github": ["git"],
    "pyspark": ["apache spark"],
    "streamlit": ["python"],
    "airflow": ["etl"],
}

RELATED_CREDIT = 0.4  # below LISTED_ONLY_CREDIT: an exact but undemonstrated claim outranks a merely related skill
IMPLIED_CREDIT = 0.9

_AMBIGUOUS = {"c", "r", "go", "node", "rest", "spring", "shell", "ts", "js", "py", "tf", "rl", "hf transformers", "ml", "swift", "express", "hive", "lambda", "s3", "ec2"}


@dataclass(frozen=True)
class Skill:
    name: str
    category: str
    aliases: tuple


SKILLS: dict[str, Skill] = {}
for _row in _RAW:
    SKILLS[_row[0]] = Skill(_row[0], _row[1], tuple(_row[2:]))

CATEGORY = {k: v.category for k, v in SKILLS.items()}

_DELIM_BEFORE = r"(?:^|[\s,;:|/•·()\[\]-])"
_BOUNDARY_L = r"(?<![A-Za-z0-9+#.])"
_BOUNDARY_R = r"(?![A-Za-z0-9+#]|\.[A-Za-z0-9])"


def _alias_pattern(alias: str) -> str:
    """Escape ``alias`` but let spaces and hyphens match each other ("machine-learning")."""
    return r"[\s\-]+".join(re.escape(tok) for tok in re.split(r"[\s\-]+", alias) if tok)


@lru_cache(maxsize=1)
def _compiled():
    """One alternation regex for unambiguous surface forms + a separate set for ambiguous ones."""
    surface = {}
    for sk in SKILLS.values():
        for form in (sk.name, *sk.aliases):
            surface[form.lower()] = sk.name
    plain = sorted((f for f in surface if f not in _AMBIGUOUS), key=len, reverse=True)
    pat = _BOUNDARY_L + "(" + "|".join(_alias_pattern(f) for f in plain) + ")" + _BOUNDARY_R
    return surface, re.compile(pat, re.I)


# Ambiguous short names are only accepted in list-like context to avoid e.g. "C" the grade
# or "Go" the verb. "node", "rest", "spring", "shell" etc. need to look like technology mentions.
_AMBIG_RE = {
    "c": re.compile(r"(?:(?<=^)|(?<=[,;|/•·:]\s)|(?<=[,;|/•·:]))\s*C(?=\s*(?:[,;|/•·]|$|\)))", re.M),
    "r": re.compile(r"(?:(?<=^)|(?<=[,;|/•·:]\s)|(?<=[,;|/•·:]))\s*R(?=\s*(?:[,;|/•·]|$|\)))", re.M),
    "go": re.compile(r"(?:(?<=^)|(?<=[,;|/•·:]\s)|(?<=[,;|/•·:]))\s*Go(?=\s*(?:[,;|/•·]|$|\)))", re.M),
    "node": re.compile(r"\bNode(?=\s*(?:[,;|/•·]|$|\)))", re.M),
    "swift": re.compile(r"\bSwift\b(?!\s+(?:response|delivery|action))"),
    "express": re.compile(r"\bExpress(?=\s*(?:[,;|/•·]|$|\))|\s*(?:\.js|js)\b|\s+(?:framework|server|api))", re.M),
    "rest": re.compile(r"\bREST\b"),
    "spring": re.compile(r"\bSpring\b(?=\s*(?:Boot|MVC|Framework|[,;|/•·]|$))", re.M),
    "shell": re.compile(r"\b(?:Unix|Linux|Bash)?\s*Shell\b(?=\s*(?:[,;|/•·]|$|script))", re.M),
    "ml": re.compile(r"\bML\b"),
    "ts": re.compile(r"(?<![A-Za-z0-9])TS(?![A-Za-z0-9])"),
    "js": re.compile(r"(?<![A-Za-z0-9])JS(?![A-Za-z0-9])"),
    "tf": re.compile(r"(?<![A-Za-z0-9])TF(?![A-Za-z0-9]|-IDF)"),
    "hive": re.compile(r"\bHive\b(?=\s*(?:[,;|/•·]|$|\)))", re.M),
    "lambda": re.compile(r"\b(?:AWS\s+)?Lambda\b"),
    "s3": re.compile(r"\bS3\b"),
    "ec2": re.compile(r"\bEC2\b"),
    "py": re.compile(r"(?<![A-Za-z0-9])py(?![A-Za-z0-9])"),
    "rl": re.compile(r"(?<![A-Za-z0-9])RL(?![A-Za-z0-9])"),
}


def find_skills(text: str) -> list[tuple[str, str, int]]:
    """Return (canonical, surface_form, offset) for every skill mention in ``text``."""
    surface, rx = _compiled()
    hits = []
    for m in rx.finditer(text):
        form = re.sub(r"[\s\-]+", " ", m.group(1).lower())
        canon = surface.get(form) or surface.get(m.group(1).lower())
        if canon is None:  # hyphen variants like "scikit-learn"
            canon = surface.get(m.group(1).lower().replace(" ", "-"))
        if canon:
            hits.append((canon, m.group(1), m.start(1)))
    for key, rx2 in _AMBIG_RE.items():
        for m in rx2.finditer(text):
            canon = surface.get(key)
            if canon:
                hits.append((canon, m.group(0).strip(), m.start()))
    return hits


def skill_set(text: str) -> set[str]:
    return {c for c, _, _ in find_skills(text)}


def expand_implied(skills: set[str]) -> dict[str, str]:
    """Map implied skill -> the skill that implies it (excluding already-present skills)."""
    out = {}
    for s in skills:
        for imp in IMPLIES.get(s, []):
            if imp not in skills and imp not in out:
                out[imp] = s
    return out


def related_skills(skill: str) -> set[str]:
    rel = set()
    for members in FAMILIES.values():
        if skill in members:
            rel.update(members)
    rel.discard(skill)
    return rel
