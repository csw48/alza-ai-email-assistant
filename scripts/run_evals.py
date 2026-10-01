"""Eval sady proti skutočnému Gemini (RAG vyhľadávanie + kvalita odpovedí).

Použitie (Vertex AI s prihláseným gcloud):
    N8N_API_KEY=x GOOGLE_CLOUD_PROJECT=moj-projekt python scripts/run_evals.py --gcloud-token
Použitie (AI Studio):
    N8N_API_KEY=x GEMINI_API_KEY=... python scripts/run_evals.py
Voľby: --only rag|replies, --thinking LOW, --model gemini-3.6-flash
"""
import argparse
import base64
import json
import mimetypes
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from google import genai  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.rag.knowledge_base import KnowledgeBase, split_queries  # noqa: E402
from app.services.gemini_service import GeminiService  # noqa: E402

EVALS = Path(__file__).resolve().parents[1] / "evals"


def build_service(args) -> GeminiService:
    cfg = settings.model_copy(update={k: v for k, v in {
        "GEMINI_MODEL": args.model, "GEMINI_THINKING_LEVEL": args.thinking,
    }.items() if v})
    if args.gcloud_token:
        from google.oauth2.credentials import Credentials

        token = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
        client = genai.Client(vertexai=True, project=cfg.GOOGLE_CLOUD_PROJECT, location=cfg.GOOGLE_CLOUD_LOCATION,
                              credentials=Credentials(token))
        return GeminiService(client, cfg.model_copy(update={"GOOGLE_GENAI_USE_VERTEXAI": True}))
    return GeminiService.from_settings(cfg)


def eval_rag(kb: KnowledgeBase) -> float:
    cases = json.loads((EVALS / "rag_cases.json").read_text())
    ok = 0
    print(f"\n=== RAG ({kb.mode}, top_k={settings.RAG_TOP_K}, min_score={settings.RAG_MIN_SCORE}) ===")
    # Všetky dotazy jedným dávkovým volaním – nové projekty majú nízku kvótu embed requestov za minútu
    vectors = None
    if kb.mode == "embeddings":
        per_case = [split_queries(c["q"]) for c in cases]
        flat = kb.embedder([q for qs in per_case for q in qs], "RETRIEVAL_QUERY")
        vectors, pos = [], 0
        for qs in per_case:
            vectors.append(flat[pos:pos + len(qs)])
            pos += len(qs)
    all_scores = []
    for i, c in enumerate(cases):
        if vectors is not None:
            all_scores.append([s for _, s in kb.rank_multi(vectors[i], len(kb.documents), 0.0, per_query=len(kb.documents))])
            results = kb.rank_multi(vectors[i], settings.RAG_TOP_K, settings.RAG_MIN_SCORE)
        else:
            results = kb.retrieve(c["q"], settings.RAG_TOP_K, settings.RAG_MIN_SCORE)
        got = [d["source"] for d, _ in results]
        top = results[0][1] if results else 0.0
        passed = all(e in got for e in c["expect"]) if c["expect"] else not got
        ok += passed
        print(f"{'PASS' if passed else 'FAIL'}  top={top:.3f}  {c['q'][:55]:55}  -> {sorted(set(got))}")
    print(f"RAG: {ok}/{len(cases)}")
    if vectors is not None:
        # Kalibrácia prahu: najvyššie skóre otázok mimo KB vs. skóre očakávaných dokumentov
        neg = [s[0] for s, c in zip(all_scores, cases) if not c["expect"]]
        print(f"Otázky mimo KB: najvyššie skóre {max(neg):.3f}  |  prah RAG_MIN_SCORE={settings.RAG_MIN_SCORE}")
    return ok / len(cases)


def eval_replies(app) -> float:
    cases = json.loads((EVALS / "reply_cases.json").read_text())
    ok, times = 0, []
    print("\n=== Odpovede ===")
    with TestClient(app) as client:
        for c in cases:
            body = {"message_id": f"<eval-{c['id']}>", "sender": "zakaznik@priklad.sk", "recipient": "bot@priklad.sk",
                    "subject": c["subject"], "body_text": c["body"], "attachments": [
                        {"filename": Path(f).name, "mime_type": mimetypes.guess_type(f)[0] or "application/octet-stream",
                         "data_base64": base64.b64encode((EVALS.parent / f).read_bytes()).decode()}
                        for f in c.get("attachments", [])]}
            time.sleep(args.delay)  # šetrí embed kvótu (1 dotaz na e-mail)
            t = time.perf_counter()
            r = client.post("/api/v1/process-email", json=body, headers={"x-api-key": settings.N8N_API_KEY})
            times.append(time.perf_counter() - t)
            if r.status_code != 200:
                print(f"FAIL  {c['id']}: HTTP {r.status_code}")
                continue
            d = r.json()
            text = d["reply_body_text"].lower()
            errors = []
            if "language" in c and d["language"] != c["language"]:
                errors.append(f"jazyk {d['language']}")
            if "needs_human" in c and d["needs_human_review"] != c["needs_human"]:
                errors.append(f"needs_human={d['needs_human_review']} ({d['review_reason']})")
            for group in c.get("must_contain_any", []):
                if not any(s.lower() in text for s in group):
                    errors.append(f"chýba {group}")
            for s in c.get("must_not_contain", []):
                if s.lower() in text:
                    errors.append(f"obsahuje '{s}'")
            ok += not errors
            print(f"{'PASS' if not errors else 'FAIL'}  {c['id']:18} {times[-1]:5.1f}s  {'; '.join(errors)}")
            if d["attachments_skipped"]:
                errors.append(f"preskočené prílohy {d['attachments_skipped']}")
            if errors and args.verbose:
                print("      " + d["reply_body_text"].replace("\n", "\n      "))
    print(f"Odpovede: {ok}/{len(cases)}, priemerný čas {sum(times) / len(times):.1f}s, max {max(times):.1f}s")
    return ok / len(cases)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gcloud-token", action="store_true", help="Vertex AI s tokenom z `gcloud auth print-access-token`")
    p.add_argument("--only", choices=["rag", "replies"])
    p.add_argument("--thinking", help="MINIMAL / LOW / MEDIUM / HIGH")
    p.add_argument("--model")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--delay", type=float, default=0.0, help="Pauza medzi e-mailmi (s), kvôli kvóte embeddingov")
    args = p.parse_args()

    ai = build_service(args)
    kb = KnowledgeBase(embedder=ai.embed)
    kb.build_index()
    if args.only != "replies":
        eval_rag(kb)
    if args.only != "rag":
        eval_replies(create_app(kb=kb, ai=ai))
