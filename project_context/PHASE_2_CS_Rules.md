# Phase 2: Rule-Based CS Detection Layer

**Owner**: Part G  
**Status**: 🔲 TODO  
**Prerequisite**: None (can run in parallel with Phase 1)  
**Outputs**: `cs_rules.py`, `features_with_cs.csv`, `cs_rule_evaluation.csv`

---

## Goal
Write explicit Snort-style CS rules for known attack signatures.
These rules are domain knowledge — not learned from data.

## Rules to Implement
- [ ] CS Rule 1: DDoS check (SYN rate > 1000/sec)
- [ ] CS Rule 2: Port scan check (unique dst ports > 100/sec)
- [ ] CS Rule 3: Small packet flood (pkt size < 60 AND rate > 500/sec)
- [ ] CS Rule 4: Exploit check (payload near MTU + URG flags)
- [ ] CS Rule 5: Backdoor check (long-duration, low-rate, non-standard port)
- [ ] CS Rule 6: Shellcode check (very small bytes, non-HTTP port)
- [ ] CS Rule 7: Worm check (high unique dst IPs from same src — proxy)

## Evaluation
- [ ] Per-rule precision, recall, F1 against true labels
- [ ] False positive rate on Benign traffic per rule
- [ ] Document which attack classes have no CS rule coverage

## Completion Criteria
- All rules implemented in `src/cs_rules/rules.py`
- `cs_rule_evaluation.csv` exported to `outputs/`
- `features_with_cs.csv` exported to `shared/outputs/`
