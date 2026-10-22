import sys
from pathlib import Path

# 1. FIX THE IMPORT PATH
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

import os
import json
import pytest

try:
    from langchain_classic.retrievers import ContextualCompressionRetriever
except ImportError:
    from langchain.retrievers import ContextualCompressionRetriever

from app.reranker import JinaReranker

# DeepEval imports
from deepeval import assert_test
from deepeval.metrics import ContextualRelevancyMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM

from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage

# ---------------------------------------------------------
# 1. ENVIRONMENT SETUP
# ---------------------------------------------------------
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "dummy")
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from app.config import settings
from app.embedding import embed_model
from app.vectorstore import pinecone_manager

# ---------------------------------------------------------
# 2. CUSTOM LLM JUDGE SETUP (UPSTAGE SOLAR-MINI4)
# ---------------------------------------------------------
class UpstageJudgeModel(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatUpstage(
            model="solar-mini4",
            temperature=0,          # CHANGED: was 0.7 — judges must be deterministic
            max_tokens=16384,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke([HumanMessage(content=prompt)]).content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke([HumanMessage(content=prompt)])
        return response.content

    def get_model_name(self):
        return "Upstage Solar Mini 4 (Custom Judge)"

custom_judge = UpstageJudgeModel()

# ---------------------------------------------------------
# 3. RETRIEVERS — TWO LAYERS FOR DIAGNOSIS
# ---------------------------------------------------------
DIAGNOSTIC_K = 10   # upper bound: "does the chunk exist in the index at all?"
PRODUCTION_K = 3    # what actually reaches the LLM after Jina reranking

vector_store = pinecone_manager.get_vector_store(embedding_model=embed_model.embedding_model)

base_retriever = vector_store.as_retriever(search_kwargs={"k": DIAGNOSTIC_K})
compressor = JinaReranker(top_n=PRODUCTION_K)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

def retrieve_both(query: str):
    """One Pinecone call + one rerank call. Returns (docs_k10, docs_k3)."""
    docs_k10 = base_retriever.invoke(query)
    try:
        docs_k3 = compressor.compress_documents(docs_k10, query)
    except Exception:
        # Fallback if your JinaReranker signature differs (costs 1 extra query)
        docs_k3 = retriever.invoke(query)
    return docs_k10, docs_k3

# ---------------------------------------------------------
# 4. DEEPEVAL METRICS (used only by the LLM-judge test)
# ---------------------------------------------------------
relevancy_metric = ContextualRelevancyMetric(threshold=0.7, model=custom_judge)
recall_metric = ContextualRecallMetric(threshold=0.7, model=custom_judge)

# ---------------------------------------------------------
# 5. LOADING THE GOLDEN DATASET (switch via env var)
# ---------------------------------------------------------
DATASET_FILE = os.getenv("GOLDEN_DATASET_FILE", "Golden_Retriver.json")
dataset_path = current_dir.parent / "golden_dataset" / DATASET_FILE

with open(dataset_path, "r", encoding="utf-8") as file:
    raw_json_data = json.load(file)

golden_dataset = raw_json_data.get("test_cases", [])
if not golden_dataset:
    raise ValueError(f"❌ No 'test_cases' found in {DATASET_FILE}.")

print(f"✅ Loaded {len(golden_dataset)} cases from {DATASET_FILE} "
      f"(scoring_mode: {raw_json_data.get('scoring_mode', 'not specified')})")

# Split by scoring strategy:
#   - Cases WITH 'expected_evidence_keywords' -> free deterministic gate (no LLM)
#   - Cases WITHOUT (e.g. the advanced trap set) -> LLM judge
keyword_cases = [d for d in golden_dataset if d.get("expected_evidence_keywords")]
judge_cases = [d for d in golden_dataset if not d.get("expected_evidence_keywords")]
print(f"   → {len(keyword_cases)} keyword-gate cases | {len(judge_cases)} LLM-judge cases")

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def _missing_keywords(keywords, docs):
    joined = "\n".join(d.page_content for d in docs).lower()
    return [kw for kw in keywords if kw.lower() not in joined]

def _doc_sources(docs):
    return [str(d.metadata.get("source", d.metadata.get("file_path", d.metadata)))[:80] for d in docs]

# ---------------------------------------------------------
# 6a. TEST 1 — DETERMINISTIC KEYWORD GATE (free, no LLM)
# ---------------------------------------------------------
@pytest.mark.keyword_gate
@pytest.mark.parametrize("data", keyword_cases, ids=[d["id"] for d in keyword_cases])
def test_keyword_gate(data):
    keywords = data["expected_evidence_keywords"]
    docs_k10, docs_k3 = retrieve_both(data["question"])

    missing_at_10 = _missing_keywords(keywords, docs_k10)
    missing_at_3 = _missing_keywords(keywords, docs_k3)

    if missing_at_10:
        pytest.fail(
            f"[{data['id']} / {data.get('tier', '?')}] ❌ VECTOR STORE MISS "
            f"(evidence not in top-{DIAGNOSTIC_K}). Missing: {missing_at_10}. "
            f"Top chunks came from: {_doc_sources(docs_k10[:3])}"
        )

    if missing_at_3:
        pytest.fail(
            f"[{data['id']}] ⚠️ RERANKER DROPPED EVIDENCE (present at k={DIAGNOSTIC_K}, "
            f"gone from top-{PRODUCTION_K}). Missing: {missing_at_3}. "
            f"Vector store is fine — consider raising JinaReranker(top_n=...)."
        )

# ---------------------------------------------------------
# 6b. TEST 2 — LLM JUDGE (only for cases without keyword gates)
# ---------------------------------------------------------
@pytest.mark.llm_judge
@pytest.mark.parametrize("data", judge_cases, ids=[d["id"] for d in judge_cases])
def test_llm_judge(data):
    if "Abstention" in data.get("test_type", ""):
        pytest.skip("Negative control — inspect manually, judge metrics don't apply.")

    retrieved_docs = retriever.invoke(data["question"])
    retrieved_context = "\n\n".join(d.page_content for d in retrieved_docs)

    test_case = LLMTestCase(
        input=data["question"],
        actual_output=retrieved_context,
        expected_output=data["ground_truth"],
        retrieval_context=[d.page_content for d in retrieved_docs],
    )
    assert_test(test_case, [relevancy_metric, recall_metric])