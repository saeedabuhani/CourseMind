"""
Regression test for the TypeError that hit a real user document:

    TypeError: '<' not supported between instances of 'NoneType' and 'NoneType'

Trigger: a document with MORE THAN ONE CHUNK ON THE SAME PAGE, answered in a
structural retrieval mode (page / overview) where distance is None. The earlier
test PDF had exactly one chunk per page, so the dedup comparison never ran.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ingestion.chunking import ChunkData
from ingestion.embeddings import embed_chunks
from ingestion.vectorstore import save_chunks, get_document_chunks, get_client, get_collection
from services import answer_question
from services.qa_service import _build_sources
from agent.tools.rag_retrieval_tool import RetrievedChunk, retrieve_page, retrieve_overview

DOC = "multichunk_regression_fixture.pdf"
# 2 pages, 4 chunks -> pages 1 and 2 each hold TWO chunks (the user's shape).
TEXTS = [
    (1, "תרגיל כיתה 3: משתנים ולולאות. בשאלה הראשונה נדרש להגדיר משתנה בשם counter ולאתחל אותו לאפס."),
    (1, "בהמשך העמוד הראשון מופיעה שאלה שנייה: כתבו לולאת for שמדפיסה את המספרים מ-1 עד 10 בעזרת console.log."),
    (2, "עמוד שני של התרגיל: פונקציות. הגדירו פונקציה בשם sum שמקבלת שני מספרים ומחזירה את הסכום שלהם."),
    (2, "בסוף העמוד השני יש שאלת בונוס על מערכים: השתמשו במתודה push כדי להוסיף איבר לסוף המערך."),
]

failures = []


def check(name, ok, detail=""):
    print("  [%s] %s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        failures.append(name)


print("1. Unit-level repro: _build_sources with two distance=None chunks on one page")
structural = [
    RetrievedChunk(text="a", source_filename=DOC, page_number=1, chunk_index=0,
                   document_id=DOC, token_count=10, character_count=40, distance=None),
    RetrievedChunk(text="b", source_filename=DOC, page_number=1, chunk_index=1,
                   document_id=DOC, token_count=10, character_count=40, distance=None),
]
try:
    srcs = _build_sources(structural)
    check("no TypeError, deduped to one citation for the page",
          len(srcs) == 1 and srcs[0].page_number == 1,
          "-> %d source(s), chunk_index=%s" % (len(srcs), srcs[0].chunk_index if srcs else "-"))
except TypeError as e:
    check("no TypeError", False, "-> STILL RAISES: %s" % e)

print("2. Mixed semantic chunks on one page still pick the lowest distance")
semantic = [
    RetrievedChunk(text="far", source_filename=DOC, page_number=1, chunk_index=0,
                   document_id=DOC, token_count=10, character_count=40, distance=1.30),
    RetrievedChunk(text="near", source_filename=DOC, page_number=1, chunk_index=1,
                   document_id=DOC, token_count=10, character_count=40, distance=0.90),
]
srcs = _build_sources(semantic)
check("best (lowest-distance) chunk wins", len(srcs) == 1 and srcs[0].chunk_index == 1,
      "-> chunk_index=%s distance=%s" % (srcs[0].chunk_index, srcs[0].distance))

print("3. End-to-end on a real indexed 2-page / 4-chunk document")
chunks = [
    ChunkData(text=t, source_filename=DOC, page_number=pg, chunk_index=i,
              document_id=DOC, token_count=max(1, len(t) // 4), character_count=len(t))
    for i, (pg, t) in enumerate(TEXTS)
]
save_chunks(chunks, embed_chunks(chunks))
stored = get_document_chunks(DOC)
per_page = {}
for c in stored:
    per_page[c["page_number"]] = per_page.get(c["page_number"], 0) + 1
check("fixture really has multiple chunks per page",
      all(v > 1 for v in per_page.values()), "-> chunks per page: %s" % per_page)

r = retrieve_page(DOC, 1)
check("retrieve_page returns both chunks of page 1", r.found and len(r.chunks) == 2,
      "-> %d chunks" % len(r.chunks))
r = retrieve_overview(DOC)
check("retrieve_overview spans both pages", r.found and len({c.page_number for c in r.chunks}) == 2,
      "-> pages %s" % sorted({c.page_number for c in r.chunks}))

print("4. The exact user flow that crashed")
try:
    resp = answer_question("תסביר לי את התוכן מה יש בקובץ?", document_id=DOC)
    check("broad/overview question answers without crashing",
          resp.found_evidence and resp.retrieval_mode == "overview",
          "-> mode=%s sources=%d pages=%s" % (resp.retrieval_mode, len(resp.sources),
                                              [s.page_number for s in resp.sources]))
except Exception as e:
    check("broad/overview question", False, "-> %s: %s" % (type(e).__name__, e))

try:
    resp = answer_question("מה יש בעמוד 1?", document_id=DOC)
    check("page question answers without crashing",
          resp.found_evidence and resp.retrieval_mode == "page"
          and all(s.page_number == 1 for s in resp.sources),
          "-> mode=%s sources=%s" % (resp.retrieval_mode, [s.page_number for s in resp.sources]))
except Exception as e:
    check("page question", False, "-> %s: %s" % (type(e).__name__, e))

try:
    resp = answer_question("מה זה לולאת for?", document_id=DOC)
    check("normal semantic question still works",
          resp.found_evidence, "-> sources=%s" % [s.page_number for s in resp.sources])
except Exception as e:
    check("normal semantic question", False, "-> %s: %s" % (type(e).__name__, e))

# clean up the fixture so it does not linger in the student's vector store
get_collection(get_client()).delete(where={"document_id": DOC})
print("\nfixture removed from the vector store")
print("RESULT: %s" % ("ALL PASS" if not failures else "FAILURES: %s" % failures))
sys.exit(1 if failures else 0)
