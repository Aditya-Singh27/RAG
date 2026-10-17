# Labeling guide (write your own notes in the `notes` column)

You see: a question, a reference answer written from the source documents, and a candidate answer. Judge the candidate only.

PASS (human_score 4 or 5)
  - the key facts agree with the reference answer, with no contradicted or invented claims
  - it answers what was asked (a refusal is a FAIL when the reference answer exists)
  - any format/length instruction in the question is respected
FAIL (human_score 1, 2 or 3)
  - a key fact is wrong, missing, or invented; or it does not answer the question; or it breaks an explicit constraint
human_score (optional but recommended): 5 fully correct | 4 correct, small omission | 3 borderline, lean FAIL | 2 mostly wrong | 1 wrong
Rules: label in one sitting if you can; do not look at key.json, any judge output, or the judge's earlier scores; if unsure, pick
the side your first reading suggested and write why in `notes`. Fill human_pass with exactly PASS or FAIL.
