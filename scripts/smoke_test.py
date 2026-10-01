"""Pošle testovací e-mail (s voliteľnými prílohami) na nasadené API a vypíše odpoveď.

Použitie:
    python scripts/smoke_test.py --url https://SERVICE.run.app --key TVOJ_N8N_API_KEY \
        --body "Koľko stojí doručenie dronom s AlzaPlus+?" foto.jpg faktura.pdf nahravka.m4a
"""
import argparse
import base64
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Base URL služby, napr. https://xxx.run.app")
    parser.add_argument("--key", required=True, help="Hodnota N8N_API_KEY")
    parser.add_argument("--subject", default="Reklamácia kávovaru a doručenie dronom")
    parser.add_argument(
        "--body",
        default="Dobrý deň, kávovar mi tečie, posielam prílohy. Môžem ho vymeniť cez AlzaBox "
        "a koľko by stálo doručenie nového kusu dronom, keď mám AlzaPlus+?",
    )
    parser.add_argument("files", nargs="*", help="Prílohy (jpg, png, pdf, mp3, wav, m4a…)")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    with urllib.request.urlopen(f"{base}/health", timeout=60) as r:
        print("health:", json.loads(r.read()))

    attachments = []
    for f in args.files:
        path = Path(f)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        attachments.append(
            {"filename": path.name, "mime_type": mime, "data_base64": base64.b64encode(path.read_bytes()).decode()}
        )

    payload = {
        "message_id": f"<smoke-{int(time.time())}@test.local>",
        "thread_id": "smoke-thread",
        "sender": "zakaznik@priklad.sk",
        "recipient": "assistant@test.local",
        "subject": args.subject,
        "body_text": args.body,
        "attachments": attachments,
    }
    req = urllib.request.Request(
        f"{base}/api/v1/process-email",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-api-key": args.key},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}")
        return 1

    print(f"\n--- odpoveď za {time.perf_counter() - start:.1f} s ---")
    for k in ("intent", "language", "needs_human_review", "review_reason", "attachments_processed",
              "attachments_skipped", "retrieved_knowledge_sources", "retrieval_mode", "processing_time_seconds"):
        print(f"{k}: {data.get(k)}")
    print("\n" + data["reply_body_text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
