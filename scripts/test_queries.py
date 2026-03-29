"""
Test script: Run 10+ diverse queries through the KisanSathi pipeline.

Covers different crops, regions, income levels, land sizes, and edge cases.

Usage:
    python scripts/test_queries.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import KisanSathiPipeline

TEST_QUERIES = [
    # --- Hindi queries (diverse topics) ---
    ("PM-KISAN योजना के लिए कौन पात्र है?", "hi", "PM-KISAN eligibility"),
    ("फसल बीमा कैसे करवाएं?", "hi", "Crop insurance enrollment"),
    ("किसान क्रेडिट कार्ड पर ब्याज दर क्या है?", "hi", "KCC interest rate"),
    ("सोलर पंप के लिए सरकारी सब्सिडी कितनी है?", "hi", "PM-KUSUM solar subsidy"),
    ("मिट्टी की जांच कैसे कराएं?", "hi", "Soil Health Card testing"),
    # --- English queries (diverse topics) ---
    ("How can I sell my crops on eNAM?", "en", "eNAM marketplace"),
    ("What is the Agriculture Infrastructure Fund?", "en", "AIF details"),
    ("Tell me about organic farming schemes", "en", "Paramparagat Krishi"),
    ("How to get drip irrigation subsidy?", "en", "PM-KISAN Sinchayee"),
    ("What schemes are available for oil palm cultivation?", "en", "NMEO oil palm"),
    # --- Edge cases ---
    ("hello", "en", "Edge: greeting/short input"),
    ("What is the weather today?", "en", "Edge: off-topic query"),
    ("2 हेक्टेयर जमीन है, कौन सी योजना मिलेगी?", "hi", "Edge: land-size specific"),
]


def run_tests():
    print("Initializing KisanSathi pipeline...\n")
    pipeline = KisanSathiPipeline()

    results = []
    for query, lang, description in TEST_QUERIES:
        print(f"{'='*70}")
        print(f"TEST: {description}")
        print(f"Query [{lang}]: {query}")
        print(f"{'='*70}")

        start = time.time()
        try:
            result = pipeline.text_query(query, lang=lang)
            elapsed = time.time() - start

            print(f"Answer: {result['answer'][:200]}...")
            print(f"Grounding: {result['grounding']['score']:.2f} "
                  f"({'GROUNDED' if result['grounding']['is_grounded'] else 'LOW'})")
            if result["sources"]:
                print(f"Top source: {result['sources'][0]['scheme']} "
                      f"(score: {result['sources'][0]['score']:.3f})")
            else:
                print("No sources (no relevant scheme found)")
            if result.get("no_result_reason"):
                print(f"No-result reason: {result['no_result_reason']}")
            print(f"Time: {elapsed:.1f}s")

            results.append({"desc": description, "status": "OK", "time": elapsed,
                            "grounded": result["grounding"]["is_grounded"]})
        except Exception as e:
            elapsed = time.time() - start
            print(f"ERROR: {e}")
            results.append({"desc": description, "status": f"ERROR: {e}", "time": elapsed,
                            "grounded": False})
        print()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    ok = sum(1 for r in results if r["status"] == "OK")
    grounded = sum(1 for r in results if r["grounded"])
    avg_time = sum(r["time"] for r in results) / len(results) if results else 0
    print(f"Passed: {ok}/{len(results)}")
    print(f"Grounded: {grounded}/{len(results)}")
    print(f"Avg response time: {avg_time:.1f}s")
    print()
    for r in results:
        status = "PASS" if r["status"] == "OK" else "FAIL"
        grnd = "G" if r["grounded"] else "-"
        print(f"  [{status}] [{grnd}] {r['desc']} ({r['time']:.1f}s)")


if __name__ == "__main__":
    run_tests()
