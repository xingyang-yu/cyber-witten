"""Cyber-Witten evaluation harness.

Measures whether the generation step stays *grounded*: every claim cited to a
passage that was actually retrieved, and graceful failure when the corpus does
not support an answer. The automated half (citation grounding, retrieval
recall) lives here; the physics half (correctness, faithfulness) is scored by a
domain expert against evals/gold/gold_set.jsonl.
"""
