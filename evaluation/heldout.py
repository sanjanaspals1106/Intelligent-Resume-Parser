"""HELD-OUT synthetic evaluation set (independent of the development generator in synthetic.py).

Purpose: provide a final test that was NOT used to develop the parser, choose the scoring weights or select the
final configuration. It differs from the development set in vocabulary, phrasing, layouts, job descriptions,
candidate archetypes and the labelling rubric:

* Candidates hold *abilities* (not taxonomy skills) at a *depth* (professional / project / coursework). Some
  abilities are realised with tools that are deliberately NOT in the system's skill taxonomy (JAX, Flink, Nomad…),
  so transferable skills are tested honestly.
* Labels come from a written rubric (``grade``) applied to latent ability facts: essential abilities dominate, a
  missing essential caps the grade, project-level evidence counts almost as much as professional evidence,
  claimed-but-not-held skills (keyword stuffing) count for nothing.
* Archetypes: exact fit, paraphrased, project-strong/limited experience, near-miss, keyword stuffing,
  transferable/alternative tooling & qualifications, ambiguous/contradictory/missing information, unrelated.

Everything is synthetic and fictional. It is NOT real-world validation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .synthetic import FIRST, LAST, MONTHS, render, render_docx, render_pdf, unicode_fonts  # noqa: F401  (renderers/names only)

SEED = 20240611        # round 1 (now the VALIDATION set: used for failure analysis)
TEST_SEED = 20250217   # round 2 (the untouched TEST set: new candidates, new layouts L5/L6)
TODAY = (2025, 6)

# ability id: (label, keyword surfaces ["Surface" or "Surface=canonical"], equivalent tools, paraphrases)
AB = {
    "py": ("Python programming", ["Python", "Python 3=python"], [], ["Wrote production libraries and command-line tools in a dynamic language favoured by scientists", "Automated recurring workflows with an interpreted high-level scripting language"]),
    "sql": ("SQL querying", ["SQL", "PostgreSQL", "MySQL"], ["Oracle PL/SQL=pl/sql"], ["Pulled and aggregated records from relational tables using joins and window functions", "Tuned slow analytical queries against a relational database"]),
    "ml": ("Machine learning modelling", ["machine learning", "scikit-learn", "XGBoost"], [], ["Fitted predictive models on tabular data and compared algorithms with cross-validation", "Engineered input variables and tuned gradient-boosted predictors for churn forecasting"]),
    "dl": ("Deep learning frameworks", ["PyTorch", "TensorFlow", "Keras"], ["JAX", "MXNet"], ["Trained multi-layer neural networks on GPUs and tuned architectures and optimisers", "Implemented custom layers and training loops for image models"]),
    "nlp": ("Natural language processing", ["NLP=natural language processing", "natural language processing", "BERT=transformers"], [], ["Built systems that classify and extract entities from free-text customer messages", "Adapted pretrained language encoders to domain-specific documents"]),
    "mlops": ("ML deployment / MLOps", ["MLflow=mlops", "MLOps"], ["Kubeflow=mlops", "BentoML"], ["Versioned experiments and automated retraining and release of models", "Served models behind monitored endpoints with automated rollbacks"]),
    "cloud": ("Cloud platforms", ["AWS", "Azure", "GCP"], ["Alibaba Cloud", "DigitalOcean"], ["Provisioned managed compute, storage and queues on a public cloud provider", "Migrated on-premise workloads to rented infrastructure"]),
    "containers": ("Containers", ["Docker", "containerization=docker"], ["Podman"], ["Packaged applications into portable images that ran identically in every environment", "Wrote image build recipes and composed multi-service local environments"]),
    "k8s": ("Container orchestration", ["Kubernetes", "k8s=kubernetes"], ["Nomad", "OpenShift"], ["Operated clusters that schedule, scale and self-heal containerised workloads", "Wrote declarative manifests for rolling updates and autoscaling"]),
    "iac": ("Infrastructure as code", ["Terraform", "Ansible=terraform"], ["Pulumi", "CloudFormation"], ["Described infrastructure in version-controlled templates that create environments on demand", "Replaced manual server setup with repeatable declarative provisioning"]),
    "cicd": ("CI/CD pipelines", ["CI/CD", "Jenkins=ci/cd", "GitHub Actions=ci/cd"], ["TeamCity", "Bamboo"], ["Built automated pipelines that test and release every merge", "Removed manual release steps with push-button deployments"]),
    "linux": ("Linux administration", ["Linux"], ["FreeBSD"], ["Administered servers from the command line, managing services, users and permissions", "Diagnosed performance problems on production machines using system tools"]),
    "monitor": ("Monitoring & alerting", ["Prometheus=monitoring", "Grafana=monitoring"], ["Nagios", "Zabbix"], ["Created dashboards and alerts so incidents were detected before customers noticed", "Defined service-level objectives and on-call runbooks"]),
    "api": ("API design", ["REST API", "FastAPI", "Django"], ["gRPC"], ["Designed HTTP endpoints with versioned JSON contracts consumed by mobile and web clients", "Documented and secured service interfaces for partner integrations"]),
    "test": ("Automated testing", ["unit testing", "pytest=unit testing", "JUnit=unit testing"], ["RSpec"], ["Wrote automated checks with mocks that run on every commit", "Raised coverage of critical modules and removed flaky checks"]),
    "msg": ("Messaging / streaming", ["Kafka"], ["RabbitMQ", "Pulsar"], ["Processed real-time event streams through a distributed log with consumer groups", "Decoupled services using durable queues"]),
    "java": ("Java / JVM", ["Java", "Spring Boot"], ["Kotlin=kotlin"], ["Built server components on the JVM using dependency injection", "Maintained enterprise services written in an object-oriented compiled language"]),
    "js": ("JavaScript / TypeScript", ["JavaScript", "TypeScript"], [], ["Wrote browser-side code that updates pages without reloads", "Added static types to a large front-end codebase"]),
    "react": ("Component UI frameworks", ["React", "Next.js", "Redux=react"], ["Svelte", "SolidJS"], ["Built component-driven single-page applications with client-side state", "Delivered server-rendered pages with file-based routing"]),
    "css": ("HTML / CSS", ["HTML", "CSS", "Tailwind=css"], [], ["Implemented responsive page layouts and accessible markup", "Styled interfaces from design tokens across breakpoints"]),
    "design": ("Interface design & prototyping", ["Figma"], ["Sketch", "Penpot"], ["Produced clickable prototypes and shared component libraries for design review", "Redesigned flows after review sessions with engineers"]),
    "research": ("User research", ["user research"], ["contextual inquiry"], ["Interviewed users and synthesised findings into personas and journey maps", "Ran moderated sessions to find usability problems"]),
    "spark": ("Distributed data processing", ["Spark=apache spark", "PySpark=apache spark"], ["Flink", "Dask"], ["Processed very large datasets in parallel across a compute cluster", "Rewrote slow single-machine jobs to run on a distributed engine"]),
    "etl": ("Data pipelines / ETL", ["ETL", "data pipelines=etl", "Airflow"], ["Luigi", "NiFi"], ["Built scheduled jobs that extract, transform and load data with retries and alerts", "Orchestrated dependent batch workflows with backfills"]),
    "dwh": ("Data warehousing", ["Snowflake=data warehousing", "BigQuery=data warehousing", "data warehousing"], ["ClickHouse"], ["Modelled star schemas in a columnar analytical store", "Optimised warehouse tables for reporting workloads"]),
    "bi": ("BI dashboards", ["Power BI", "Tableau"], ["Superset", "QuickSight"], ["Built interactive management dashboards for non-technical stakeholders", "Replaced static reports with self-service visual analytics"]),
    "excel": ("Spreadsheet modelling", ["Excel"], ["Google Sheets"], ["Maintained spreadsheet models with lookups, pivots and macros", "Reconciled monthly figures in workbooks used by finance"]),
    "stats": ("Statistics & experimentation", ["statistics", "hypothesis testing=statistics", "A/B testing=statistics"], [], ["Designed experiments and interpreted significance and confidence intervals", "Modelled variability to separate signal from noise"]),
    "agile": ("Agile delivery", ["Agile", "Scrum=agile", "Kanban=agile"], [], ["Ran sprint ceremonies and backlog grooming for a cross-functional team", "Coached the team to ship in small increments"]),
    "pm": ("Project management", ["project management", "stakeholder management=project management", "Jira=jira"], ["Asana", "MS Project"], ["Planned timelines, budgets and risks and reported progress to sponsors", "Coordinated deliverables across teams and managed dependencies"]),
}

# job id: title, essential, important, nice, min_years, text
JOBS = [
    {"id": "H1", "ess": ["py", "ml", "dl"], "imp": ["sql", "mlops", "cloud"], "nice": ["containers"], "years": 2, "text": """Machine Learning Engineer, Applied Research Team
What you will do
- Prototype, train and ship models that power our ranking products
What we need
- 2+ years building machine learning systems in Python
- Hands-on experience with a deep learning framework (PyTorch, TensorFlow or similar)
- Comfortable with SQL and with deploying models on a cloud platform
Would be nice
- Docker, MLflow
"""},
    {"id": "H2", "ess": ["nlp", "py"], "imp": ["dl", "ml"], "nice": ["mlops"], "years": 3, "text": """NLP Scientist

We are an early-stage startup looking for a language-technology specialist with at least three years of experience.
You will build natural language processing models for contract analysis, so strong Python and real NLP experience are a must.
Familiarity with deep learning and general machine learning methods is expected, and experience putting models into production would be a bonus.
"""},
    {"id": "H3", "ess": ["sql", "bi"], "imp": ["excel", "stats"], "nice": ["py"], "years": 1, "text": """Data Analyst (Commercial)
Requirements:
* 1+ years analysing business data
* Confident writing SQL
* Dashboarding experience in Power BI or Tableau
* Good grasp of statistics; advanced Excel
Preferred: Python for analysis.
"""},
    {"id": "H4", "ess": ["py", "api", "sql"], "imp": ["containers", "test", "cloud"], "nice": ["msg"], "years": 3, "text": """Backend Engineer - Python
Responsibilities
- Own services that expose REST APIs to our mobile apps
Requirements
- 3+ years of Python backend development
- Experience designing and securing REST APIs
- Solid SQL / relational database skills
- Docker, automated testing and a public cloud
Nice to have: Kafka
"""},
    {"id": "H5", "ess": ["java", "api", "sql"], "imp": ["test", "msg"], "nice": ["containers", "cloud"], "years": 4, "text": """Senior Java Engineer
Qualifications:
- 4+ years of Java (Spring Boot)
- RESTful API design, relational databases
- Test automation and messaging systems such as Kafka
Preferred: Docker, any cloud
"""},
    {"id": "H6", "ess": ["js", "react", "css"], "imp": ["test", "design"], "nice": ["api"], "years": 2, "text": """Frontend Engineer
You bring:
- 2+ years building web UIs with JavaScript/TypeScript
- React or a comparable component framework
- Strong HTML and CSS
- Testing habits and the ability to work from Figma designs
Bonus: experience integrating REST APIs
"""},
    {"id": "H7", "ess": ["containers", "k8s", "cicd"], "imp": ["iac", "linux", "cloud"], "nice": ["monitor"], "years": 3, "text": """Platform / DevOps Engineer
Must have
- 3+ years running production infrastructure
- Docker and Kubernetes (or equivalent orchestration)
- CI/CD pipeline design
- Infrastructure-as-code (Terraform or similar), Linux and a major cloud
Plus: monitoring and observability tooling
"""},
    {"id": "H8", "ess": ["py", "sql", "etl"], "imp": ["spark", "dwh", "cloud"], "nice": ["msg"], "years": 3, "text": """Data Engineer
Responsibilites (sic):
- Build reliable data pipelines feeding our analytics warehouse
Requirments:
- 3+ yrs with Python and SQL
- Experience designing ETL / ELT processes
- Spark or another distributed engine, cloud data warehouses
Nice: streaming (Kafka)
"""},
    {"id": "H9", "ess": ["pm", "agile"], "imp": ["excel", "bi"], "nice": ["sql"], "years": 4, "text": """Technical Project Manager
Requirements:
- 4+ years managing software projects
- Agile delivery (Scrum or Kanban) and strong stakeholder management
- Budget tracking in Excel; reporting dashboards
- Bachelor's degree or equivalent experience
Nice to have: SQL
"""},
    {"id": "H10", "ess": ["design", "research"], "imp": ["css"], "nice": ["js"], "years": 2, "text": """Product / UX Designer
- 2+ years designing digital products
- Prototyping in Figma (or similar tools)
- Planning and running user research
- Working knowledge of HTML and CSS
Helpful: some JavaScript
"""},
]
DEPTH_W = {"pro": 1.0, "project": 0.8, "course": 0.4}
ROLE_TITLES = {
    "H1": ["ML Engineer", "Applied Scientist"], "H2": ["NLP Engineer", "Language Technology Researcher"], "H3": ["Data Analyst", "Insights Analyst"], "H4": ["Backend Developer", "Software Engineer"],
    "H5": ["Java Developer", "Software Engineer"], "H6": ["Frontend Developer", "Web Engineer"], "H7": ["DevOps Engineer", "Platform Engineer"], "H8": ["Data Engineer", "ETL Developer"],
    "H9": ["Project Manager", "Delivery Lead"], "H10": ["UX Designer", "Product Designer"],
}
UNRELATED = [("Retail Supervisor", ["Managed shift rosters for a team of twelve", "Handled stock counts and supplier orders", "Resolved customer complaints at the service desk"]),
             ("Accounts Assistant", ["Prepared supplier invoices and bank reconciliations", "Maintained the fixed-asset register", "Supported quarterly audits"]),
             ("Nurse Educator", ["Delivered clinical training sessions for new staff", "Updated care protocols after incident reviews", "Mentored student nurses on placement"])]
COMPANIES = ["Greywood Systems", "Tidewater Analytics", "Kestrel Software Ltd", "Maple & Finch Consulting", "Ironbridge Digital", "Solstice Labs", "Harrow Technologies", "Brightpath Solutions", "Ardent Cloud Inc", "Lakeside Data Co"]
UNIS = ["Ashford University", "Brookfield Institute of Technology", "Coastal State College", "Dunmore University", "Elmwood College", "Fairhaven Institute", "Glenview University"]
DEGS = [("BSc", "Computer Science", 3), ("BEng", "Software Engineering", 3), ("BSc", "Mathematics", 3), ("MSc", "Data Science", 4), ("BA", "Design", 3), ("BBA", "Business Administration", 3), ("MSc", "Physics", 4), ("PhD", "Statistics", 5), ("Diploma", "Information Technology", 2)]
BOOT = "Nextstep Coding Bootcamp"
CERTS = [("AWS Certified Developer", "Amazon Web Services"), ("Certified ScrumMaster", "Scrum Alliance"), ("Google Data Analytics Certificate", "Coursera"), ("Kubernetes Application Developer", "Linux Foundation")]


def _parse(tok):
    return (tok.split("=")[0], tok.split("=")[1].lower() if "=" in tok else tok.lower())


@dataclass
class Cand:
    cid: str
    target: str
    archetype: str
    layout: str
    name: str
    email: str
    phone: str
    held: dict  # ability -> depth
    claimed: set
    years: float
    gold_skills: set = field(default_factory=set)
    jobs: list = field(default_factory=list)
    education: list = field(default_factory=list)
    projects: list = field(default_factory=list)
    certs: list = field(default_factory=list)
    structure: list = field(default_factory=list)
    messy: str = ""

    def truth(self):
        return {"name": [self.name], "email": [self.email], "phone": ["".join(c for c in self.phone if c.isdigit())], "skills": sorted(self.gold_skills),
                "exp_title": [j[0] for j in self.jobs], "exp_org": [j[1] for j in self.jobs],
                "edu_institution": [e[2] for e in self.education], "edu_level": [f"L{e[3]}" for e in self.education],
                "project_name": [p[0] for p in self.projects], "cert_name": [c[0] for c in self.certs]}


def grade(c: Cand, job: dict) -> int:
    """Rubric (see module docstring). Uses only latent facts."""
    w = lambda a: DEPTH_W.get(c.held.get(a), 0.0)
    cov = lambda ids: sum(w(a) for a in ids) / len(ids) if ids else 1.0
    e, i, n = cov(job["ess"]), cov(job["imp"]), cov(job["nice"])
    score = 0.6 * e + 0.3 * i + 0.1 * n
    g = 3 if score >= 0.8 else 2 if score >= 0.55 else 1 if score >= 0.3 else 0
    weakest = min(w(a) for a in job["ess"])
    if weakest == 0:
        g = min(g, 1)          # an essential ability that is entirely absent caps relevance
    elif weakest < 0.5:
        g = min(g, 2)          # coursework-only essential
    pro_ess = sum(c.held.get(a) == "pro" for a in job["ess"]) / len(job["ess"])
    if c.years < 0.5 * job["years"] and pro_ess < 0.5 and g > 0 and e < 0.7:
        g -= 1                 # short experience matters only when essentials are not shown (professionally or via projects)
    return g


def _realise(rng, ab, named: bool, equiv: bool, verb_ctx: str):
    label, kws, eqs, paras = AB[ab]
    if named:
        surf, canon = _parse(rng.choice(eqs if equiv and eqs else kws))
        return f"{rng.choice(['Used', 'Applied', 'Worked with', 'Delivered features using', 'Relied on'])} {surf} {verb_ctx}.", surf, canon
    return rng.choice(paras) + ".", None, None


def make(rng: random.Random, idx: int, target: dict, arch: str, layout: str) -> Cand:
    ess, imp, nice = target["ess"], target["imp"], target["nice"]
    allk = list(AB)
    held, claimed, named_p, equiv_set = {}, set(), 0.9, set()
    years = rng.choice([target["years"], target["years"] + 1, target["years"] + 3])
    if arch == "exact":
        held = {a: "pro" for a in ess + imp}
        held.update({a: "pro" for a in nice if rng.random() < .5})
    elif arch == "paraphrased":
        held = {a: "pro" for a in ess + imp}
        named_p = 0.35
    elif arch == "project":
        held = {a: "project" for a in ess + imp}
        pro = rng.choice(ess + imp)
        if rng.random() < .5:
            held[pro] = "pro"
        years = rng.choice([0, 0.5, 1])
    elif arch == "near_miss":
        miss = rng.choice(ess)
        held = {a: "pro" for a in ess + imp if a != miss}
        years = target["years"] + rng.choice([0, 2])
    elif arch == "stuffed":
        held = {rng.choice(imp + nice or ess): "pro"}
        claimed = set(ess + imp + nice)
        years = rng.choice([0, 1, 2])
    elif arch == "transferable":
        held = {a: "pro" for a in ess + imp}
        equiv_set = set(rng.sample([a for a in ess + imp if AB[a][2]] or ess, min(2, len([a for a in ess + imp if AB[a][2]]) or 1)))
    elif arch == "ambiguous":
        held = {a: "pro" for a in ess + imp if rng.random() < .85}
        held.setdefault(ess[0], "pro")
    else:  # unrelated
        other = rng.choice([j for j in JOBS if j["id"] != target["id"]])
        held = {a: "pro" for a in rng.sample(other["ess"] + other["imp"], 2)}
        years = rng.choice([1, 3, 5])
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    slug = "".join(ch for ch in name.lower() if ch.isalpha() or ch == " ").replace(" ", "_")
    c = Cand(f"H{idx:03d}", target["id"], arch, layout, name, f"{slug}{idx}@heldout-example.org", f"+44 7{rng.randint(100, 999)} {rng.randint(100000, 999999)}", held, claimed, years)
    titles = ROLE_TITLES[target["id"]]
    ey, em = TODAY
    n_jobs = 0 if years == 0 else 1 if years <= 2 else 2
    ends, spans = ey * 12 + em - 1, []
    for j in range(n_jobs):
        per = max(3, int(years * 12 / n_jobs))
        s_ = ends - per
        spans.append(((s_ // 12, s_ % 12 + 1), None if j == 0 else (ends // 12, ends % 12 + 1)))
        ends = s_ - rng.randint(1, 3)
    pro_abs = [a for a, d in held.items() if d == "pro"]
    proj_abs = [a for a, d in held.items() if d == "project"]
    st = c.structure
    H = {"L1": lambda t: t, "L2": lambda t: t.upper(), "L3": lambda t: t, "L4": lambda t: t, "L5": lambda t: t.upper(), "L6": lambda t: t}[layout]
    heads = {"L1": ("Profile", "Experience", "Education", "Projects", "Certifications", "Skills"),
             "L2": ("About", "Career History", "Qualifications", "Selected Work", "Certificates", "Key Skills"),
             "L3": ("Summary", "Professional Experience", "Education and Training", "Personal Projects", "Licenses and Certifications", "Technical Skills"),
             "L4": ("My story", "What I have done", "Where I studied", "Things I built", "Courses", "Tools I like"),
             "L5": ("Profile", "Experience", "Education", "Projects", "Certifications", "Skills"),
             "L6": ("▪ PROFILE", "▪ EMPLOYMENT", "▪ EDUCATION", "▪ SELECTED PROJECTS", "▪ CERTIFICATIONS", "▪ SKILLS")}[layout]
    hh = lambda i: ("h", H(heads[i]) if layout not in ("L1", "L6") else heads[i])
    COLS = {0: "R", 1: "R", 2: "L", 3: "R", 4: "L", 5: "L"}
    bul = lambda ln: ("line", "- " + ln) if layout == "L6" else ("bullet" if layout != "L2" else "line", ln)
    messy_noheads = arch == "ambiguous" and rng.random() < .5
    missing_dates = arch == "ambiguous" and not messy_noheads
    contradictory = arch == "ambiguous" and rng.random() < .6
    use_head = (lambda i: hh(i)) if not messy_noheads else (lambda i: ("blank", ""))
    if layout == "L5":
        _uh = use_head
        use_head = lambda i: [("col", COLS[i]), _uh(i)]
    def _add(x):
        st.extend(x) if isinstance(x, list) else st.append(x)

    st.append(("name", name))
    st.append(("contact", f"{c.email}  ·  {c.phone}"))
    named_skill_pool = []

    def bullets(abs_, ctx, equiv_ok=True):
        out = []
        for a in abs_:
            line, surf, canon = _realise(rng, a, rng.random() < named_p, a in equiv_set and equiv_ok, ctx)
            if surf:
                c.gold_skills.add(canon)
                named_skill_pool.append(surf)
            out.append(line)
        return out

    rolemix = pro_abs[:]
    rng.shuffle(rolemix)
    if arch == "stuffed" or arch == "unrelated":
        ut, ub = rng.choice(UNRELATED)
    _add(use_head(0))
    st.append(("line", f"{rng.choice(titles) if arch not in ('stuffed', 'unrelated') else ut} with {int(years)} years' experience." + (" Passionate team player, fast learner, results-driven." if arch == "stuffed" else "")))
    _add(use_head(1))
    for ji in range(n_jobs):
        s_, e_ = spans[ji]
        ttl = rng.choice(titles) if arch not in ("stuffed", "unrelated") else ut
        org = rng.choice(COMPANIES)
        dfmt = (lambda d: f"{MONTHS[d[1] - 1]} {d[0]}") if layout in ("L1", "L3") else (lambda d: f"{d[1]:02d}/{d[0]}") if layout == "L5" else (lambda d: f"{d[0]}")
        dr = f"{dfmt(s_)} {'–' if layout != 'L3' else 'to'} {dfmt(e_) if e_ else ('Present' if layout != 'L3' else 'Current')}"
        if contradictory and ji == 0:
            dr = f"{dfmt(e_ or (ey, em))} – {dfmt(s_)}"   # end before start
        if missing_dates:
            dr = ""
        head = f"{ttl}, {org}" + (f"  ({dr})" if dr else "") if layout in ("L1", "L4") else f"{org} | {ttl}" + (f" | {dr}" if dr else "")
        if layout == "L5":
            head = f"{ttl} — {org}" + (f" ({dr})" if dr else "")
        if layout == "L6":
            st.append(("line", f"{org} — {ttl}"))
            head = dr
        if head:
            st.append(("line", head))
        c.jobs.append((ttl, org, spans[ji]))
        chunk = rolemix[ji::max(1, n_jobs)] if arch not in ("stuffed", "unrelated") else []
        lines = bullets(chunk, "in production") if chunk else []
        if arch in ("stuffed", "unrelated"):
            lines += ub
        if arch == "unrelated" and rolemix:
            lines += bullets(rolemix, "when the role needed it")
        for ln in lines:
            st.append(bul(ln + ("" if ln.endswith(".") else ".")))
        st.append(("blank", ""))
    if n_jobs == 0:
        st.append(("line", "No professional experience yet; see projects."))
    _add(use_head(2))
    deg = rng.choice(DEGS)
    nodeg = arch in ("project", "transferable") and rng.random() < .4
    if arch == "transferable":
        deg = rng.choice([d for d in DEGS if d[1] in ("Physics", "Mathematics", "Statistics", "Business Administration", "Design")])
    ys = ey - int(years) - 4
    if nodeg:
        st.append(("line", f"{BOOT} - Full-stack and data certificate ({ys + 3})"))
        c.education.append((None, None, BOOT, 2))
    else:
        uni = rng.choice(UNIS)
        if layout == "L6":
            st += [("line", uni), ("line", f"{deg[0]} in {deg[1]} | {ys} - {ys + 3}")]
        else:
            st.append(("line", f"{deg[0]} {deg[1]}, {uni}, {ys}–{ys + 3}" if layout != "L2" else f"{uni} — {deg[0]} in {deg[1]} ({ys}–{ys + 3})"))
        c.education.append((deg[0], deg[1], uni, deg[2]))
    if proj_abs or arch in ("exact", "paraphrased", "transferable") and rng.random() < .6:
        _add(use_head(3))
        pa = proj_abs or rng.sample(pro_abs, min(2, len(pro_abs)))
        for k in range(0, len(pa), 2):
            nm = rng.choice(["Harbor", "Lumen", "Cobalt", "Tern", "Quill", "Vesper"]) + " " + rng.choice(["Planner", "Tracker", "Hub", "Atlas", "Forge"]) + f" {k // 2 + 1}"
            st.append(("line", nm))
            for ln in bullets(pa[k:k + 2], "as a personal project"):
                st.append(bul(ln))
            c.projects.append((nm,))
            st.append(("blank", ""))
    course = [a for a, d in held.items() if d == "course"]
    if rng.random() < .35 and arch not in ("stuffed", "unrelated"):
        ce = rng.choice(CERTS)
        _add(use_head(4))
        st.append(("line", f"{ce[0]} - {ce[1]}, {rng.randint(2020, 2024)}"))
        c.certs.append(ce)
    # skills list
    listed = []
    for a in (sorted(claimed) if arch == "stuffed" else []):  # sorted: set order would depend on PYTHONHASHSEED
        surf, canon = _parse(rng.choice(AB[a][1]))
        listed.append(surf)
        c.gold_skills.add(canon)
    if arch not in ("stuffed", "unrelated"):
        for surf in dict.fromkeys(named_skill_pool):
            listed.append(surf)
    if listed and rng.random() < .85 or arch == "stuffed":
        _add(use_head(5))
        rng.shuffle(listed)
        st.append(("line", ("Skills: " if layout != "L4" else "") + ", ".join(dict.fromkeys(listed))))
    return c


def build(seed=SEED, per_arch=None, layouts=("L1", "L2", "L3", "L4")):
    rng = random.Random(seed)
    arch = per_arch or [("exact", 2), ("paraphrased", 2), ("project", 2), ("near_miss", 2), ("stuffed", 1), ("transferable", 2), ("ambiguous", 2), ("unrelated", 1)]
    cands, idx = [], 0
    for job in JOBS:
        for a, n in arch:
            for _ in range(n):
                idx += 1
                cands.append(make(rng, idx, job, a, layouts[idx % len(layouts)]))
    return cands, JOBS


def render_two_column_pdf(structure) -> bytes:
    """Name/contact across the top, then 'L' sections in a narrow left column and 'R' sections in a wide right column."""
    import fitz
    import textwrap

    top = [x for x in structure if x[0] in ("name", "contact")]
    left, right = [], []
    cur = right
    for kind, text in structure:
        if kind == "col":
            cur = left if text == "L" else right
        elif kind not in ("name", "contact"):
            cur.append((kind, text))
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    uf = unicode_fonts()
    if uf:
        page.insert_font(fontname="uni", fontfile=uf[0])
        page.insert_font(fontname="unib", fontfile=uf[1])
    reg, bold = ("uni", "unib") if uf else ("helv", "hebo")
    y = 56
    for kind, text in top:
        page.insert_text((50, y), text, fontname=bold if kind == "name" else reg, fontsize=18 if kind == "name" else 9.5)
        y += 26 if kind == "name" else 14
    for col, items, x, width in ((0, left, 40, 26 if uf else 30), (1, right, 235, 54 if uf else 62)):
        yy = y + 14
        for kind, text in items:
            if kind == "blank":
                yy += 6
                continue
            size = 11.5 if kind == "h" else 9.5
            if kind == "h":
                yy += 8
            for ln in textwrap.wrap(("• " if kind == "bullet" else "") + text, width, subsequent_indent="  " if kind == "bullet" else "") or [""]:
                if yy > 800:
                    break
                page.insert_text((x, yy), ln, fontname=bold if kind == "h" else reg, fontsize=size)
                yy += size + 3
    if uf:
        try:
            doc.subset_fonts()
        except Exception:
            pass
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return out


def render_cand(c: Cand):
    if c.layout == "L3":
        return f"{c.cid}.docx", render_docx(c.structure)
    if c.layout == "L5":
        return f"{c.cid}.pdf", render_two_column_pdf(c.structure)
    return f"{c.cid}.pdf", render_pdf([x for x in c.structure if x[0] != "col"], unicode_font=True)
