# Non-AI/ML Domain Practices for Improving CoT Performance

> Compiled 2026-05-08. Companion to `CROSS_DISCIPLINARY_IDEAS.md` (which still lives in
> stat / info-theory / CS-adjacent territory). This file is the one specifically asked
> for: **borrows from fields that have nothing to do with machine learning** — medicine,
> aviation, law, manufacturing, military, surgery, performing arts, games, journalism,
> diplomacy.
>
> **Selection rule**: each entry must be a *codified practice* (not just an analogy) from
> a non-AI domain that has a non-trivial mapping to an aspect of CoT-CP or CoT performance
> in general. The goal is concrete prompt-engineering, scaffolding, or evaluation
> protocols inspired by how *humans* in high-stakes domains discipline reasoning.

---

## Why this file

LLM Chain-of-Thought is, in operational terms, *humans-doing-careful-reasoning translated
into auto-regressive tokens*. Every codified human protocol for careful reasoning under
pressure — checklists, after-action reviews, root-cause analysis, adversarial review — is
prior art for prompt engineering and reasoning-trace evaluation. The CoT literature (and
CoT-CP itself) reinvents these practices ad hoc. This file catalogs them properly,
maps each onto a CoT-CP design choice, and proposes concrete experimental adoptions.

For each entry: **Source · Practice · CoT-CP mapping · Concrete proposal · Cost/Value**.

---

# A. Medicine & clinical reasoning

## A1. Differential Diagnosis (DDx) and the VINDICATE mnemonic

- **Source**: Bickley, L. S. (2020), *Bates' Guide to Physical Examination and History Taking*, 13th ed., Wolters Kluwer. Standard medical-school curriculum, several centuries of practice.
- **Practice**: When presented with a symptom, a physician *enumerates* an ordered list of plausible causes (the differential), then narrows by evidence (pertinent positives/negatives) to the working diagnosis. Mnemonics like **VINDICATE** (Vascular, Infectious/Inflammatory, Neoplastic, Drugs/Degenerative, Iatrogenic, Congenital, Autoimmune, Trauma, Endocrine) force breadth before depth. Critical: the differential is *explicitly written down*, not implicit.
- **CoT-CP mapping**: Today's CoT generation is **single-thread** — the model commits to one solution path immediately. A "DDx-style CoT" would force the model to enumerate K candidate solution paths *before* choosing one, with explicit pertinent-evidence-for / against each. This is structurally different from naive K-resample (Pilot C/K/L) because the K paths are generated *with knowledge of each other* — the model has to argue against the others. Should reduce the cascade-failure mode where one early wrong commitment locks in.
- **Concrete proposal**: Add a "DDx prompt" before the solution: *"Before solving, list 3 candidate solution strategies: (1) ... (2) ... (3) ... For each, state the strongest reason it might be the right approach and the strongest reason it might fail. Then commit to one and proceed."* Score each generated trace by whether the chosen strategy survived the model's own opposition.
- **Connects to**: Layer B step rejection (Pilot C/K/L null result — DDx may give the structural diversity that K-resample failed to provide); §6 future-work; new Wave-5 prompt experiment.
- **Cost/Value**: 2-day prompt experiment on MATH-500 / AIME-2024. Value: high if it changes the K=4-majority null result; even a 2-3pp absolute lift would be meaningful.

## A2. Clinical concurrence rule (two-physician sign-off for high-stakes orders)

- **Source**: Joint Commission (2014), *National Patient Safety Goals*, NPSG.03.04.01 (high-alert medication double-check). Codified post-1999 IOM "To Err Is Human" report.
- **Practice**: Certain irreversible or high-risk clinical actions (chemotherapy doses, blood transfusions, pediatric weight-based meds) require two independent verifications before execution. The second verifier is *blind* to the first's reasoning to prevent confirmation bias.
- **CoT-CP mapping**: Self-consistency at N=2 is *not* a concurrence rule because both samples come from the same model with the same prompt — they are correlated. Real concurrence requires (a) different prompts (paraphrase), (b) different temperatures, (c) blinded comparison. Closest existing CoT work: paraphrase consensus M=3 (in our Pilot M, Wave 1 agent_D) gives MATH-500 92.7% / AIME 79.6%, comparable to SC@8. The medical practice formalizes *blindness* as the key — the second look should not see the first.
- **Concrete proposal**: Implement "blinded concurrence" as a 2-step pipeline: (1) generate a candidate solution with prompt P; (2) hide the candidate, generate a *fresh* solution with paraphrased prompt P'; (3) accept iff both arrive at the same answer. Compare to ordinary SC@2 (which is correlated). Likely much stronger because samples are *independent*.
- **Connects to**: SC family on the Layer A score ladder; could become a new score `concur_blind` between `lp_min` (1×) and `prm_min` (2×) at cost ~2.5×.
- **Cost/Value**: 1 week experiment, 4-6 hours compute. Value: med-high — fills a cost gap on our Pareto if it dominates SC@2.

## A3. SBAR communication (Situation, Background, Assessment, Recommendation)

- **Source**: Leonard, M., Graham, S., Bonacum, D. (2004), *The human factor: the critical importance of effective teamwork and communication in providing safe care*, BMJ Quality & Safety. Adopted from US Navy nuclear submarine practice.
- **Practice**: Structured 4-part handoff in healthcare. Information must be presented in this order; the *Recommendation* is required (no passive observation). Eliminates the most common failure mode: a nurse reports symptoms but doesn't say what they think should happen.
- **CoT-CP mapping**: A CoT trace is essentially a handoff to the answer-extractor: the trace should communicate enough for an oracle reader to verify. Today's CoT is unstructured prose; SBAR-style would force: *Situation* (what's the problem), *Background* (what assumptions / knowns), *Assessment* (what calculation / argument), *Recommendation* (the answer + confidence). Forces explicit confidence statement at the end — which is exactly what we want for `prov_match_latest_rate`-style scoring.
- **Concrete proposal**: Add an explicit "Confidence" field at the end of each step: *"Step N: [reasoning]. Confidence in this step: [LOW / MED / HIGH] because [...]"*. The model's verbalized confidence is then a per-step score `s_t = verbalized_conf` directly available without logits. Test whether this verbal-confidence outperforms or correlates with `lp_min`. Connect to Tian-Mitchell (EMNLP 2023) "Just Ask for Calibration" — verbalized confidence is poor by default but improvable.
- **Connects to**: New score family — verbalized step confidence; closed-source-API friendly (no logits needed).
- **Cost/Value**: 1 week. Value: med — likely correlated with `lp_min` but may extend to API-only models.

## A4. Morbidity & Mortality (M&M) conferences

- **Source**: Codman, E. A. (1916), *A Study in Hospital Efficiency*. Standard practice in US/UK/JP hospitals; AMA Code of Medical Ethics requires participation. See Pierluissi+ (2003), *Discussion of medical errors in morbidity and mortality conferences*, JAMA.
- **Practice**: Public, structured retrospective on adverse outcomes — explicitly framed as "what went wrong, and how do we prevent it?" rather than "who is to blame." Required attendance; presenting physician (often a resident) walks through the case; senior physicians question the reasoning at every step. Generates institutional-memory documents.
- **CoT-CP mapping**: This is a *post-hoc* protocol — applies to traces that produced wrong answers. Today's CoT failure analysis is ad-hoc ("PRM said this step was bad"). M&M-style analysis would: (a) collect failed traces with ground truth, (b) decompose each failure into which step *first* deviated, (c) classify the deviation type (arithmetic, retrieval, planning, premature commitment, garden-path, etc.), (d) produce a taxonomy. Resulting failure-mode taxonomy is the *evaluation target* for new score functions.
- **Concrete proposal**: Run M&M-style review on 50-100 failed AIME traces. Build a classification of dominant failure modes. Use the taxonomy to: (i) target Wave-5 score-design at the most common failure modes (currently we score what's easy, not what fails most); (ii) report a per-failure-mode kept-accuracy in §5 (e.g., "CoT-CP at α=0.5 catches 73% of arithmetic-error traces and 41% of premature-commitment traces"). Per-failure-mode evaluation is what radiology AI papers do; CoT papers do not.
- **Connects to**: §5 evaluation breakdown; Wave-5 score design; ProcessBench-style benchmarking.
- **Cost/Value**: 2-3 days human review time (or LLM-as-judge with careful spec). Value: very high — turns aggregate accuracy into structured diagnostic.

## A5. Triage scoring (APACHE-II, GCS, NEWS2)

- **Source**: Knaus, W. A. et al. (1985), *APACHE II: A severity of disease classification system*, Critical Care Medicine. Teasdale, G. & Jennett, B. (1974), *Assessment of coma and impaired consciousness — A practical scale*, Lancet (GCS).
- **Practice**: A *fixed-formula* numeric score combining 12-15 vital signs + lab values into a single severity number that triages care intensity. Calibrated against mortality outcomes from large registries. Crucially: the score is *not* the diagnosis, only a routing signal.
- **CoT-CP mapping**: Our `tiebreak_lex` and weighted score combinations are essentially uncalibrated APACHE-style fusions. The discipline lesson is: (a) calibrate the score against an *outcome* (mortality / wrong-answer) on a large held-out cohort with bootstrap CIs (we already do this — it's CP); (b) *use the score for routing only, not for diagnosis* — the score is whether to abstain / branch, not whether the answer is right. We sometimes blur this. Strengthen the conceptual separation in §3.
- **Connects to**: §3 framing; weighted-score combination ablation.
- **Cost/Value**: 0 (writing decision). Value: low-med — sharpens the framework.

---

# B. Aviation & aerospace safety

## B1. Cockpit checklist (Gawande's *Checklist Manifesto*)

- **Source**: Gawande, A. (2009), *The Checklist Manifesto: How to Get Things Right*, Metropolitan Books. WHO Surgical Safety Checklist (2008). Original aviation precedent: Boeing B-17 1935 crash → standardized pre-flight checklist.
- **Practice**: A *paper checklist* run aloud by two crew members before every critical phase (engine start, takeoff, descent, landing). Items are short, action-verb-led, and *forced* — not optional. The 2008 WHO surgical checklist (a one-page, 19-item document run at three time-points) reduced surgical complications by ≥ 30% in 8 hospitals across multiple countries.
- **CoT-CP mapping**: The closest CoT analog is "few-shot exemplars," but those teach by example, not by *forced step-coverage*. A pre-decode checklist would force the model to address every standard item before producing the trace: *"[ ] Have you parsed all numerical values? [ ] Have you identified the unknown? [ ] Have you considered units? [ ] Are there special cases (zero, negative, boundary)?"* This is structurally different from CoT prompting because the items are *exhaustive*, not illustrative.
- **Concrete proposal**: Build a math-checklist from PRM800K failure-mode statistics (after running A4 above). Prepend: *"Before solving, address each of the following — say 'addressed' or write your reasoning: (1) Verify all given numerical values; (2) Identify what is being asked; (3) Note units and constraints; (4) Consider degenerate cases. Then proceed."* Compare kept-accuracy with vs without the checklist. Hypothesize the gain is on AIME (where premature commitment is common) and zero on GSM8K (already easy).
- **Connects to**: Base trace quality (improves Layer A's input); Wave-5 prompt engineering.
- **Cost/Value**: 1 week. Value: med-high — directly attacks the "premature commitment" failure mode.

## B2. Crew Resource Management (CRM) and the 2-pilot redundancy rule

- **Source**: Helmreich, R. L., Merritt, A. C., Wilhelm, J. A. (1999), *The evolution of Crew Resource Management training in commercial aviation*, International Journal of Aviation Psychology. Mandated by the FAA after the Tenerife disaster (1977) and United 173 (1978).
- **Practice**: Two pilots with explicit role separation (Pilot Flying / Pilot Monitoring). The Monitoring pilot is *required* to verbalize cross-checks ("speed alive," "rotate," "v1") and challenge the Flying pilot's actions. CRM training explicitly teaches *junior crew to challenge senior crew* — addresses the steep authority gradient that caused Korean Air 801 (1997) and Air Florida 90 (1982).
- **CoT-CP mapping**: This is the formalization of "challenge mode" — explicit, mandatory adversarial verification rather than optional self-check. Maps onto: (a) two-model pipelines where the second model is *required* to challenge the first's reasoning; (b) "verify, then trust" — trust only steps that have been challenged.
- **Concrete proposal**: 2-pass CoT: pass 1 generates the trace (Pilot Flying); pass 2 *must* find one objection per step (Pilot Monitoring). Final answer accepted iff every objection is overruled with an argument. Score each step by whether the objection was overruled or sustained. This is structurally different from CoVe (Chain-of-Verification, ACL Findings 2024 in our notes) because CRM is *mandatory* — there's no "if confident, skip" loophole.
- **Connects to**: Wave 2 agent_C external verifier; Step rejection Layer B (per-step challenge); paired-LLM architecture.
- **Cost/Value**: 1-2 weeks experimental. Value: high if the CoVe-style refinement is amplified by mandatory-objection. Cost: 2× inference.

## B3. Sterile cockpit rule (FAR 121.542 / 135.100)

- **Source**: FAA (1981), *Federal Aviation Regulations* §121.542 — no non-essential conversation below 10,000 ft.
- **Practice**: During critical flight phases, only *operationally relevant* communication is allowed. Prevents distraction-induced errors (Eastern 401 in 1972 was distracted by a burnt-out indicator light; flew into the Everglades).
- **CoT-CP mapping**: CoT traces frequently contain *narrative filler* — restating the question, social hedging ("Let me think carefully"), self-reassurance — that crowds out reasoning tokens. Especially severe in long-CoT models (R1, QwQ). Sterile-cockpit-style prompting would *forbid* filler during the reasoning phase.
- **Concrete proposal**: System prompt: *"During the solution, every sentence must contain either a fact, a computation, or a conclusion. Sentences that restate the problem, express hesitation, or comment on the difficulty are forbidden. Violations will be excerpted and reviewed."* Compare to baseline. Likely effects: shorter traces, less truncation, possibly better PRM scores (since each step is denser).
- **Connects to**: Long-CoT model issues (R1-Distill 60+ steps, often noisy); token-budget-aware generation.
- **Cost/Value**: 3-5 days. Value: med — likely small absolute lift but cleaner traces help downstream scoring.

## B4. NTSB-style probable-cause investigation

- **Source**: NTSB (1962-present), *Aviation Accident Reports*. See public archive at ntsb.gov.
- **Practice**: Every commercial aviation accident triggers an investigation that produces a *Probable Cause* statement plus *Contributing Factors* — explicitly causally distinguished. Reports include explicit recommendations (numbered, addressed, tracked across years).
- **CoT-CP mapping**: This is the gold standard for failure-mode taxonomy. Distinct from M&M (A4) in that NTSB reports are public, archival, and recommendations are *tracked* — same recommendation re-issued if a similar accident recurs. Maps onto: build an open-archive of "CoT accidents" — failed traces with structured probable-cause statements.
- **Concrete proposal**: Same as A4 (M&M) but make it public-archive style and include *recommendations* — "to prevent this failure mode, future versions should..." Becomes a paper artifact (appendix or website) rather than just a §5 figure. Has knock-on benefit for the field.
- **Connects to**: Paper appendix; potential standalone artifact (a CoT-failure benchmark).
- **Cost/Value**: 2 weeks but multi-paper value. Value: high if executed thoroughly.

---

# C. Surgery & critical procedures

## C1. WHO Surgical Time-Out (3-phase pause)

- **Source**: Haynes, A. B. et al. (2009), *A Surgical Safety Checklist to Reduce Morbidity and Mortality in a Global Population*, NEJM 360:491-499. The 19-item, 3-phase WHO checklist (sign in / time out / sign out).
- **Practice**: At three explicit pause-points — before anesthesia, before incision, before the patient leaves the OR — the entire team stops, names the item, addresses each line, and only resumes when consensus is reached. The *time-out before incision* is the most rigorously enforced: every team member states their name, role, the patient name, the procedure, the operative site (mark visible), antibiotic timing, and known allergies. This 60-second pause is the single most-cost-effective intervention in surgical safety.
- **CoT-CP mapping**: A pre-commitment pause before irreversible reasoning steps. Identifies "critical phase" reasoning operations — the algebraic substitution, the case split, the limit operation — and inserts an explicit verification block before them. Different from CoT-decoding-without-prompting (Wang-Zhou NeurIPS 2024) which is post-hoc; this is *forced* pre-commit verification.
- **Concrete proposal**: Detect critical operations (substitution, case-split, limit, integration by parts) via regex on the partial trace, and inject: *"[Verification before proceeding] Before applying [operation], verify: (a) [antecedent condition 1]; (b) [antecedent condition 2]. Confirmed? [yes/no with justification]"*. Run on Olympiad benchmarks where critical-step errors are common.
- **Connects to**: Layer B (pre-step verification); Wave 2 agent_C external verifier — more disciplined version.
- **Cost/Value**: 1 week. Value: med-high if it reduces the dominant failure mode.

## C2. Sponge / instrument count

- **Source**: Joint Commission, *NPSG 1990s*. Codified after multi-hundred-million-dollar lawsuits over retained surgical items.
- **Practice**: Every sponge, needle, and instrument is *counted* before incision and *re-counted* before closure. Discrepancy stops the procedure. The count is recorded and signed.
- **CoT-CP mapping**: A reasoning trace's "instruments" are the named variables, the given quantities, and the unknowns. Sponge-count-style verification: at the start, list all variables explicitly; at the end, verify each was used or discarded with reason. Catches the failure mode where a problem says "let A, B, C be..." and the model never uses C.
- **Concrete proposal**: Output schema enforcement: pre-trace *"Variables: x = [...], y = [...], constraint: [...]"*; post-trace *"Variable usage check: x used in step N, y used in step M, constraint applied at step K."* Fail trace if any variable is unaccounted for.
- **Connects to**: Pre/post-trace structure; new validation pass.
- **Cost/Value**: 2 days. Value: low-med — narrow but easy.

## C3. Sterile field maintenance (gowning, draping, double-glove)

- **Source**: AORN (2024), *Guidelines for Perioperative Practice*, Association of periOperative Registered Nurses.
- **Practice**: Layered redundancy against contamination. Single-glove failures are caught by the second glove; gown breaches caught by drape; etc. No single-point-of-failure.
- **CoT-CP mapping**: Layered redundancy in score family — each step gated by *multiple* score thresholds (lp_min AND prm_min AND sc_top1), not just one. We currently use `tiebreak_lex` (8× + free) which is layered. The surgical doctrine says: layer *more aggressively* on the highest-stakes steps. Could add a 4th, 5th layer at marginal cost.
- **Connects to**: `tiebreak_lex` extension; ensemble scoring.
- **Cost/Value**: 1 week. Value: low — likely diminishing returns.

---

# D. Law & adversarial systems

## D1. IRAC method (Issue, Rule, Application, Conclusion)

- **Source**: ABA / Columbia Law (2021), *Organizing a Legal Discussion (IRAC, CRAC, etc.)*, Columbia Law student writing center. See Wikipedia: IRAC. Standard since c. 1950 in US bar exam preparation.
- **Practice**: Every legal analysis follows the IRAC structure: (1) **Issue** — what is the legal question; (2) **Rule** — what statute/precedent applies; (3) **Application** — apply rule to facts; (4) **Conclusion** — answer to issue. Variants: CRAC (Conclusion first), IRAAAC (multiple Applications), TREAC (Topic-Rule-Explanation-Application-Conclusion). All enforce explicit separation of facts, law, reasoning, conclusion.
- **CoT-CP mapping**: Math reasoning today blurs problem-restatement, formula-recall, computation, and answer. IRAC-structured CoT would force: (1) **Issue** = restate what's asked; (2) **Rule** = state which theorem/identity applies; (3) **Application** = the calculation; (4) **Conclusion** = the answer. Crucially, *Rule* is verifiable via knowledge-lookup (does this theorem exist? is the form correct?), and *Application* is verifiable via re-derivation. Separation enables per-component scoring.
- **Concrete proposal**: Prompt template: *"Use the following structure exactly: ISSUE: [restate]; RULE: [name and state the theorem/identity used]; APPLICATION: [step-by-step calculation]; CONCLUSION: [boxed answer]."* Score Rule via knowledge-base lookup (is the named theorem real and correctly stated?). Score Application via PRM. Compare to free-form CoT.
- **Connects to**: Structured-output generation; per-component scoring.
- **Cost/Value**: 1 week experiment. Value: med-high — cleanly separable scoring is a paper-strong story.

## D2. Burden of proof gradations

- **Source**: McCormick on Evidence (8th ed. 2020), §339-§341. Federal Rules of Evidence.
- **Practice**: Three standards: (i) *preponderance of evidence* — civil cases, "more likely than not," ≈ 50.1% confidence; (ii) *clear and convincing* — fraud, immigration, ≈ 75% confidence; (iii) *beyond a reasonable doubt* — criminal cases, ≈ 95% confidence. The *judge* (or rules-system) picks the standard based on the stakes; the *jury* applies it.
- **CoT-CP mapping**: This is the user-facing α-grid! Different applications (high-stakes math vs casual QA vs medical reasoning) demand different α — and the *user* picks. Our paper currently treats α as a researcher-set knob; reframing it as a "burden of proof" — chosen by the user based on stakes — is a much more natural §6 framing.
- **Connects to**: §1 motivation; §6 deployment narrative.
- **Cost/Value**: 0 (framing). Value: med — improves narrative.

## D3. Daubert standard for expert testimony

- **Source**: *Daubert v. Merrell Dow Pharmaceuticals*, 509 U.S. 579 (1993). Federal Rules of Evidence Rule 702.
- **Practice**: When an expert witness offers testimony, the trial judge acts as a "gatekeeper" — they may admit the testimony only if (a) the methodology is testable / falsifiable; (b) it has been peer-reviewed; (c) it has a known error rate; (d) it is generally accepted in the relevant scientific community. Rejects "ipse dixit" (he said it, therefore it's true).
- **CoT-CP mapping**: Direct map onto how *we* should frame our paper's methodology: each conformity score (`lp_min`, `prm_min`, `sc_top1`) should pass these four tests. Daubert (a) testable: yes — we report kept-accuracy with bootstrap CIs. (b) peer-reviewed: paper review. (c) known error rate: this is *exactly* the CRC coverage statement. (d) generally accepted: tied to CP-for-LLM literature. The framework gives us a clean rebuttal to "this is just a heuristic" reviews.
- **Concrete proposal**: §3 should explicitly cast each score family as a "Daubert-passing" expert witness. Reframes the contribution as *methodological discipline* rather than *new method*.
- **Connects to**: §1 and §3 framing.
- **Cost/Value**: 0. Value: low-med — pure framing, but elegant.

## D4. Cross-examination

- **Source**: Wigmore on Evidence (1904, 4th ed. 1972), §1367: cross-examination as "the greatest legal engine ever invented for the discovery of truth." Mauet, T. A. (2017), *Trial Techniques and Trials*, 10th ed., Aspen.
- **Practice**: Adversarial system: a witness's direct testimony is followed by cross-examination from opposing counsel, who may ask leading questions and introduce contradictions. The contrast between direct and cross is what the jury weighs.
- **CoT-CP mapping**: Self-consistency without challenge is *direct testimony only* — N independent witnesses telling the same story. Self-consistency *with challenge* would force one of the N samples to play opposing counsel: explicitly attack the others' reasoning. This is structurally different from majority vote.
- **Concrete proposal**: Generate N=3 with standard prompt; generate N=1 with adversarial prompt ("find the strongest objection to this solution: ..."); accept iff the standard solutions survive the adversarial sample. Different from CoVe (Chain-of-Verification) because the adversary is *required* to attack rather than self-check.
- **Connects to**: SC family extensions; CoVe-style scaffolds.
- **Cost/Value**: 1 week. Value: med — likely incremental over SC@4 but cleaner.

---

# E. Engineering quality / process

## E1. Failure Mode and Effects Analysis (FMEA)

- **Source**: US Military (1949), *Procedures for Performing a Failure Mode, Effects and Criticality Analysis*, MIL-P-1629. Adopted by NASA Apollo, then automotive (Ford 1977 Pinto recall). ASQ (American Society for Quality) maintains modern documentation.
- **Practice**: For each component of a system, enumerate every *failure mode* (how it could fail), every *effect* (consequence at the system level), and every *cause* (why it might fail). Score each on Severity (1-10), Occurrence (1-10), Detection (1-10); product = Risk Priority Number (RPN). Address top-RPN items first. Highly proceduralized — performed in formal team review with documented sign-off.
- **CoT-CP mapping**: For CoT, "components" are prompt sections, decoding modes, score functions, and post-processing steps. FMEA produces a *prioritized failure-mode list* that drives where to invest engineering effort. We currently invest in score-function design (Wave 1-3 zoo); FMEA might say the biggest risk is *prompt-format brittleness* or *answer-extraction failures* (which our `robust_eval.py` partly addresses).
- **Concrete proposal**: Conduct an FMEA workshop on the CoT-CP pipeline. Components: prompt format, sampling, segmentation, score calculation, answer extraction. For each, list failure modes (e.g., "prompt forces wrong language," "segmentation lumps two steps as one," "answer extractor regex misses LaTeX `\\boxed`"). Score with our existing data. The exercise will *change which experiments to run* — likely surfacing infrastructure improvements that beat new score-design.
- **Connects to**: Engineering effort prioritization; §5 limitations section.
- **Cost/Value**: 1 day workshop + 2 days writing. Value: high — likely changes priorities.

## E2. Five Whys (Toyota Production System)

- **Source**: Ohno, T. (1988), *Toyota Production System: Beyond Large-Scale Production*, Productivity Press. Sakichi Toyoda originated the technique c. 1930.
- **Practice**: When a defect occurs, ask "why?" five times in succession to drill from symptom to root cause. *Example*: "the machine stopped → why? → an overload blew a fuse → why? → the bearing was insufficiently lubricated → why? → the pump wasn't pumping enough oil → why? → the pump shaft was worn → why? → no filter, so metal scrap entered the pump." The proximate fix (replace the fuse) is rejected; the root cause fix (install a filter) is taken.
- **CoT-CP mapping**: A failed CoT trace's *proximate* error is "the model wrote 7 instead of 9 at step 12." Five Whys would chase: "why 7? → because the partial sum at step 11 was wrong → why? → because step 10's binomial coefficient was misremembered → why? → because the formula was applied with the wrong index → ..." The dominant failure mode for math CoT is rarely the last step; it's almost always an upstream commitment. Our existing per-step scoring catches *worst* steps, not *first-divergent* steps. Pilot 2.1 (Earliest-bad-step re-roll) was motivated by exactly this insight, sourced from HGJ feedback.
- **Concrete proposal**: Validate "earliest-divergent step" as a re-roll target experimentally. We already planned this (HGJ Idea 2.1, +1-2 GPU hr, paper value HIGH). The Five Whys frame gives the *theoretical* justification for why earliest is more likely to be the cascade source than worst.
- **Connects to**: HGJ_experiment_ideas Idea 1.1; Layer B re-roll target choice.
- **Cost/Value**: Already planned (1-2 GPU hr). Value: HIGH (HGJ priority).

## E3. Statistical Process Control (SPC) / Shewhart charts

- **Source**: Shewhart, W. A. (1931), *Economic Control of Quality of Manufactured Product*, D. Van Nostrand. ISO 7870-2:2013, *Control charts*.
- **Practice**: Plot a process metric (e.g., dimension of a machined part) over time. Compute mean ± 3σ as control limits from the calibration period. Out-of-control rules (Western Electric / Nelson 1984): point outside 3σ; 2 of 3 outside 2σ; 6 in a row trending; etc. Signal triggers root-cause investigation.
- **CoT-CP mapping**: Per-step score distributions during decoding are a process. Western-Electric-style rules would detect "this trace is trending down" (6 consecutive steps with declining `lp_min`) before any single step crosses a hard threshold. Provides a more sensitive trigger than our per-step CP Approach A (single-threshold abort).
- **Concrete proposal**: Implement a Shewhart-style mid-trace abort: maintain running mean ± 2σ of step scores during the calibration period; abort traces that hit any of the 8 Nelson rules. Compare to single-threshold Approach A. Expected: catches *deterioration* earlier than Approach A catches *catastrophe*.
- **Connects to**: Per-step CP Approach A extension; new Layer A trigger.
- **Cost/Value**: 1 week. Value: med-high — adds a new trigger family.

## E4. Six Sigma DMAIC cycle

- **Source**: Pyzdek, T. & Keller, P. (2018), *The Six Sigma Handbook*, 5th ed., McGraw-Hill. Originated at Motorola 1986.
- **Practice**: Define-Measure-Analyze-Improve-Control. A 5-phase iterative project methodology for quality improvement. Each phase has explicit deliverables and a tollgate before moving on.
- **CoT-CP mapping**: This is the *research methodology* template for our CoT-CP improvement loop. Maps onto: Define = pick failure mode (from A4 / B4); Measure = current kept-accuracy on that subset; Analyze = root-cause via Five Whys (E2); Improve = run targeted experiment; Control = monitor regression. Already implicit in our pipelines/Wave structure but worth making explicit as a template.
- **Connects to**: Research methodology / appendix.
- **Cost/Value**: 0. Value: low — meta-methodology.

## E5. Poka-yoke (mistake-proofing) — Toyota / Shingo

- **Source**: Shingo, S. (1986), *Zero Quality Control: Source Inspection and the Poka-yoke System*, Productivity Press.
- **Practice**: Design the system so the mistake is *physically impossible*. Examples: a USB-A connector that only fits one way; a gas-tank cap tethered so it can't be left behind; a microwave that can't run with the door open.
- **CoT-CP mapping**: Build the prompt and answer-extraction such that the most common mistakes are *structurally impossible*. Examples: format the answer as JSON (impossible to forget the boxed answer); require explicit unit specification (impossible to lose units); force step-numbering (impossible to skip). The key shift is from "detect mistakes after the fact" (CP) to "make mistakes impossible up front" (Poka-yoke). They're complementary.
- **Concrete proposal**: Audit common failure modes from A4/B4 and redesign prompts to make each impossible. Example: "expected answer is an integer in [0, 999]" → enforce in prompt → answer extractor regex matches `\\boxed{[0-9]{1,3}}` only. AIME answers are 0-999 integers — this Poka-yoke alone might add 2-5pp by eliminating extraction failures.
- **Connects to**: Prompt engineering; `robust_eval.py` extension.
- **Cost/Value**: 2-3 days. Value: med — narrow but high ROI on extraction.

---

# F. Military strategic thought

## F1. OODA loop (Observe-Orient-Decide-Act, John Boyd)

- **Source**: Boyd, J. R. (c. 1976-1995), *A Discourse on Winning and Losing*, briefings; published posthumously. See Coram, R. (2002), *Boyd: The Fighter Pilot Who Changed the Art of War*, Little, Brown. The OODA loop is the most-cited concept in modern military doctrine.
- **Practice**: A decision-cycle: Observe (gather data), Orient (interpret in context, *the most important step* per Boyd), Decide (commit to action), Act (execute). The fighter pilot who completes this cycle faster than the adversary wins. Boyd's deeper claim: *Orient* is hardest because it depends on prior models, biases, and culture — and the Orient stage is where most decisions go wrong.
- **CoT-CP mapping**: A reasoning trace is a sequence of OODA loops. Today's CoT collapses Observe + Orient ("what is being asked, and what kind of problem is this?") into the first sentence. Boyd would argue: spend most of the budget on Orient — explicitly identify the problem class, the relevant theorem family, the analogous solved problems. Our current "few-shot exemplar" prompting partially does this. Strict-OODA prompting would force separation.
- **Concrete proposal**: Pre-decode prompt: *"Step 1 (Observe): list all given quantities. Step 2 (Orient): identify the problem class (algebra / number theory / combinatorics / geometry / calculus) and the most likely solution method. Step 3 (Decide): commit to a method. Step 4 (Act): execute."* Compare to free-form CoT. The hypothesis: forced Orient on AIME problems yields higher accuracy because the model is forced to recognize *which kind of problem* before solving — a step strong solvers do reflexively, weak solvers skip.
- **Connects to**: Prompt template experiments; cf. IRAC (D1).
- **Cost/Value**: 1 week. Value: med-high — directly attacks miscategorization failures.

## F2. After-Action Review (US Army FM 7-0)

- **Source**: US Army Combined Arms Center (2008), *FM 7-0 Training for Full Spectrum Operations*. AAR formalized 1981; civilian adoption (PMI, project management) widespread.
- **Practice**: Immediately after an operation, all participants gather for a structured 4-question retrospective: (1) What was supposed to happen? (2) What actually happened? (3) Why was there a difference? (4) What will we do differently next time? Recommendations are written, tracked, and reviewed in subsequent AARs to ensure closure.
- **CoT-CP mapping**: This is the M&M conference (A4) plus formal recommendation tracking. The 4-question structure is *exact* — could be the LLM-as-judge prompt for failure analysis. Apply to every wrong CoT trace in a held-out audit set; aggregate "what we will do differently" across N=200 traces to derive prompt-improvement directions.
- **Concrete proposal**: After-action LLM-judge: for each failed trace, generate (1)-(4) using a strong model. Cluster the (4)s into improvement directions. Implement top-3 improvements as new prompt or score variants.
- **Connects to**: §5 evaluation; iterative improvement methodology.
- **Cost/Value**: 1 week initial; ongoing low cost. Value: high — turns failure traces into improvement signal.

## F3. Red Team / Blue Team

- **Source**: US Army TRADOC Pam 525-7-19 (2010), *Red Teaming University Handbook*. Originated in Cold War-era CIA war-gaming; standardized in 2003 University of Foreign Military and Cultural Studies.
- **Practice**: Institutionalized devil's advocacy. A *Blue Team* designs the operation; a *Red Team* — separate, often outsourced — is required to find every failure mode and adversarial exploit. Red Team's success metric is *attacks found*, not *attacks prevented* — the incentive structures are deliberately opposed.
- **CoT-CP mapping**: Self-critique is weak because the same model has the same blind spots. A *different model* as Red Team is structurally stronger. Cross-Model Verification (CMV) protocol in our internal `cross_model_verification_protocol.md` is exactly this — `claude-opus-4-7` produces, `openai/gpt-5.5` verifies. We could push it harder: require Red-Team to *find a counter-example* on every Blue-Team-accepted answer, not just verify.
- **Concrete proposal**: For accept-α=0.5 traces, run Red Team prompt: *"Find an arithmetic, logical, or domain error in this solution. If you cannot, state explicitly that no error was found. Output: [error description] OR [no error found]."* Aggregate Red Team success rate; revise α based on it.
- **Connects to**: Cross-model verification protocol; CMV existing infrastructure.
- **Cost/Value**: 1 week. Value: high — leverages existing CMV setup.

## F4. Commander's Intent (Auftragstaktik)

- **Source**: Moltke the Elder (c. 1860); modern: US Army FM 6-0 (2014), *Commander and Staff Organization and Operations*.
- **Practice**: Orders specify the *intent* (what the operation must achieve, what are the success criteria) without prescribing the *method*. Subordinates are empowered to deviate from plan if conditions change but the intent is preserved. Famous historical proof-point: German Wehrmacht's mission-tactics doctrine (Auftragstaktik) outperforming Allied centralized command in the 1939-1942 period.
- **CoT-CP mapping**: When a CoT trace gets stuck, current practice is to either persist (most CoT) or restart (some recent reasoning models with backtrack tokens "Wait, let me reconsider..."). Auftragstaktik suggests a third path: *re-state the goal*, then re-plan locally. Maps onto: insert an "intent-restatement" token at long stalls or low-confidence points. Different from DEER's exit-trigger because the model isn't exiting, just re-grounding.
- **Concrete proposal**: When `lp_drawdown` exceeds a threshold mid-trace, inject: *"[Intent restatement: the original problem asks for X. Current sub-goal: Y.] Continue."* Gives the model an opportunity to re-orient without restarting.
- **Connects to**: Per-step CP Approach A; Layer B alternative to early-abort.
- **Cost/Value**: 1 week. Value: med — directly relevant to long-CoT models.

---

# G. Performing arts (music, theatre)

## G1. Sectional rehearsal (orchestral practice)

- **Source**: Standard practice in orchestral conducting; see Rudolf, M. (1995), *The Grammar of Conducting*, Schirmer Books.
- **Practice**: Before tutti rehearsal, each instrumental section (strings / winds / brass / percussion) rehearses separately to perfect their part. Tutti rehearsal then focuses on integration — voice leading, intonation, balance — not on note-learning.
- **CoT-CP mapping**: Current CoT trains/decodes everything end-to-end. Sectional rehearsal suggests: train (or prompt) sub-skills separately, then integrate. Maps onto curriculum learning during fine-tuning, but more directly: structure the prompt so the model first solves a *sub-problem* (e.g., simplify the algebra), confirms it, then integrates into the main problem.
- **Concrete proposal**: Sub-problem decomposition prompt: *"Step 1: Solve the algebra portion in isolation. Step 2: Verify your algebra by substitution. Step 3: Integrate into the main problem."* Compare to direct end-to-end CoT.
- **Connects to**: Wave 4 step rejection (chains hierarchically); prompt engineering.
- **Cost/Value**: 1 week. Value: low-med — many CoT prompts already do this implicitly.

## G2. Improvisation rules (jazz: II-V-I, fakebook)

- **Source**: Levine, M. (1995), *The Jazz Theory Book*, Sher Music Co. The "Real Book" (1970s) — fakebook of jazz standards.
- **Practice**: Skilled jazz improvisers don't generate notes randomly; they have memorized chord progressions (ii-V-I) and licks that they recombine on-the-fly. The constraint enables creativity, not vice versa.
- **CoT-CP mapping**: LLMs already have "memorized progressions" (theorems, identities, common transformations). Strong CoT involves *retrieving* the right progression and *fitting* the problem to it, not generating from scratch. Map onto: retrieval-augmented CoT where the retrieval is over a *lick book* of standard moves (e.g., "AM-GM inequality," "triangle inequality reverse," "telescoping sum"). Different from RAG over text — retrieval is over named *transformations*.
- **Concrete proposal**: Build a math "fakebook" of 30-50 named transformations; at each step, the model must identify which fakebook entry applies (or "none — direct calculation"). Score the trace by valid fakebook usage. Could be a §5 ablation on AIME/Olympiad.
- **Connects to**: Tool-use / RAG hybrid; structured CoT.
- **Cost/Value**: 2-3 weeks (fakebook construction is non-trivial). Value: med.

---

# H. Games & sport

## H1. Chess engine annotation (multipv, eval, !,!!,?,??)

- **Source**: Chess Informant (1966-present); modern: Stockfish, Leela. Standard PGN annotations: !! brilliant, ! good, !? interesting, ?! dubious, ? mistake, ?? blunder.
- **Practice**: Strong chess engines compute multiple principal variations (multipv = top-K continuations with eval scores) at every move. Annotations grade move quality on a 6-point scale. The combination of *parallel exploration* (multipv) + *quality annotation* is the modern analytical standard.
- **CoT-CP mapping**: Beam search with K candidates already gives multipv-style parallel exploration; what's missing is the *annotation* — explicit per-step quality scores. Our `prm_min` does this implicitly with a continuous score; chess-style discrete annotations (5-6 levels) might be more interpretable for users and more robust to PRM-noise. Also: chess multipv shows that *humans use top-3, not top-1* during analysis — we should report kept-accuracy across the top-3 traces, not just the kept-and-correct.
- **Concrete proposal**: Quantize PRM scores into 5 bins (??, ?, !?, !, !!); report per-bin accuracy. Adds an interpretable axis to §5 figures: "with PRM-score in the top bin (!!), kept-accuracy is X%; in the second bin (!), Y%."
- **Connects to**: §5 figure design; PRM score interpretation.
- **Cost/Value**: 2 days. Value: low-med — interpretation/aesthetic.

## H2. Replay review and challenge systems (NFL, tennis)

- **Source**: NFL Rule 15 (Instant Replay); tennis Hawk-Eye review system.
- **Practice**: A subset of decisions can be challenged; the challenger pays a cost (loss of timeout, loss of challenge if wrong). Resolution by independent video review. Post-challenge ruling is final. Critically: not all decisions are reviewable, and the cost of a frivolous challenge is non-zero.
- **CoT-CP mapping**: Maps onto budget-constrained re-roll: re-rolls cost compute; the system must decide which steps are *worth* challenging. Our current Tier 5 gating (`gate_combined_f0.5_K4`) does exactly this — gate via combined entropy ∨ lp ∨ arith and only re-roll the gated steps. The replay-system framing gives us:
  1. A discipline: *limit* the number of re-rolls per trace (NFL has 3 challenges per game).
  2. A penalty: failed re-rolls (where the alternative is no better) should reduce the budget for further re-rolls in the trace.
- **Concrete proposal**: Implement *N-replay budget* per trace — max 2 re-rolls allowed; if both alternatives produce the same answer as the original, lock the trace and no further re-rolls. Test if this beats unlimited gating. Likely small effect but cleaner story.
- **Connects to**: Tier 5 gating extension; budget-aware Layer B.
- **Cost/Value**: 1 week. Value: med — likely small empirical effect, clean story.

## H3. Tournament seeding (Swiss vs single-elim)

- **Source**: FIDE Handbook (2024), *Swiss System Tournament Rules*. Standard chess and bridge tournament practice.
- **Practice**: Swiss-style tournaments pair players of similar score in each round, ensuring K rounds discriminate K^2 strength levels. Single-elimination eliminates losers immediately — fast but high-variance.
- **CoT-CP mapping**: When evaluating *which score family wins on which prompt* (cross-model, cross-dataset matrix), we currently run all-pairs. Swiss-style would: rank scores after a few prompts, pair them on subsequent prompts where they disagree most, converge faster on which dominates. Sample-efficient evaluation.
- **Connects to**: Evaluation methodology (low priority for v1).
- **Cost/Value**: 2 weeks. Value: low — optimization, not contribution.

---

# I. Journalism & verification

## I1. Two-source rule

- **Source**: Kovach, B. & Rosenstiel, T. (2014), *The Elements of Journalism*, 3rd ed., Three Rivers Press. Codified at *Washington Post* during Watergate (Bradlee–Bernstein–Woodward); standard in major newsrooms.
- **Practice**: A controversial fact requires confirmation by ≥2 independent sources before publication. Sources must be *independent* — not citing each other, not from the same office, etc.
- **CoT-CP mapping**: Self-consistency at N=2 is *not* two-source — both samples come from the same model. Real two-source: (a) different model (different training data); (b) different prompt (paraphrase); (c) different decoding (greedy vs sampled). For each step, accept iff ≥2 of the 3 independent sources agree on the value. Different from majority vote because *independence* is the criterion, not just count.
- **Concrete proposal**: Implement 3-source step verification: (i) Qwen3-8B-no-think greedy; (ii) Qwen2.5-Math-7B paraphrase; (iii) DeepSeek-R1-Distill-7B sampled. Accept step iff ≥2 agree. Compare to SC@8 single-model. Cost: 3× inference; should beat single-model SC by leveraging model diversity.
- **Connects to**: Cross-model verification; Pilot M (PRM+SC ensemble was negative — independence criterion may save it).
- **Cost/Value**: 1-2 weeks. Value: med-high — leverages our multi-model infrastructure.

## I2. Inverted pyramid structure

- **Source**: Standard journalism textbook, since c. 1865 (Civil War correspondents). Ward, S. J. A. (2015), *The Invention of Journalism Ethics*, 2nd ed., McGill-Queen's UP.
- **Practice**: News article structure: most-important fact first (the lead), then progressively less critical detail. A reader can stop at any point and still have the gist.
- **CoT-CP mapping**: CoT prose buries the answer at the end. Inverted pyramid would put the answer first, then the reasoning. This is the *Conclusion-Reasoning-Application-Conclusion* (CRAC) variant of IRAC (D1). The benefit: enables truncation-aware decoding — if max_tokens is hit, the answer is still present. R1-Distill traces frequently truncate at max_tok=1536; an answer-first structure would salvage them.
- **Concrete proposal**: Prompt: *"State your answer in the form 'I believe the answer is X.' Then justify in detail."* Compare to standard CoT on long-CoT models with aggressive max_tok limits. Likely effects: graceful degradation under truncation.
- **Connects to**: Long-CoT model handling; truncation rerun (E6v3, E12r, etc.).
- **Cost/Value**: 3-5 days. Value: med — addresses a known practical issue.

## I3. Editorial chain of review

- **Source**: AP Stylebook (2024), 56th ed. Standard newspaper hierarchy: reporter → assigning editor → copy editor → managing editor → publisher.
- **Practice**: A story passes through ≥3 independent reviews before publication. Each reviewer has a different lens (factual accuracy / narrative coherence / style / legal exposure / commercial impact). The output of each is *required revision* — not optional.
- **CoT-CP mapping**: Multi-pass editing of a CoT trace, with each pass having a different prompt. Pass 1: factual / arithmetic verification. Pass 2: logical coherence. Pass 3: format / units. Pass 4: final answer extraction. Could be cheap (each pass is short) and structured.
- **Concrete proposal**: 3-pass editorial pipeline as a structured `Layer C` on top of A and B. Each pass uses the same model with a different system prompt. Compare to single-pass.
- **Connects to**: Pipeline architecture; potential v2 paper extension.
- **Cost/Value**: 2 weeks. Value: med — lots of moving parts; might be diluted.

---

# J. Diplomacy & negotiation

## J1. BATNA (Best Alternative to Negotiated Agreement)

- **Source**: Fisher, R., Ury, W., Patton, B. (1981/2011), *Getting to Yes: Negotiating Agreement Without Giving In*, 3rd ed., Penguin. Harvard Negotiation Project.
- **Practice**: Before entering a negotiation, identify your *BATNA* — what you'll do if no agreement is reached. Walk away if the offered agreement is worse than the BATNA. The BATNA bounds your concessions.
- **CoT-CP mapping**: For *abstain-or-answer* decisions, the BATNA is "abstain." The kept-accuracy threshold should be *above* the abstain alternative — which is the user's expected utility of "I don't know." For high-stakes uses (medical, legal), abstain has high value (no wrong action); for casual use, abstain has low value. The α-grid we report should be paired with a BATNA-style utility argument: "at α=0.5, kept-accuracy 79.3%; abstain saves the user from 20.7% wrong answers; if a wrong answer costs more than 4× a missed answer, choose α=0.5."
- **Connects to**: §6 deployment narrative; user-facing guidance.
- **Cost/Value**: 0 (framing). Value: low-med.

## J2. Confidence-Building Measures (CBMs, Helsinki Final Act)

- **Source**: Conference on Security and Co-operation in Europe (1975), *Helsinki Final Act*, Basket I Section II. Codified term in arms-control diplomacy.
- **Practice**: Two adversaries who don't trust each other adopt *low-cost* observable signals (military exercise notification, observer exchange) that demonstrate good faith. Each CBM has low information-leakage but high signal-of-intent. Iterated CBMs build trust enabling larger agreements.
- **CoT-CP mapping**: When two reasoning models or two prompts are used (cross-model verification), they need a "trust" protocol. CBMs map to *cheap shared verifications* — both must produce the same answer to a sub-question whose answer is independently known (a sympy-verifiable algebra fragment, a known fact). Failure of a CBM check disqualifies the trace from acceptance regardless of other scores.
- **Connects to**: Cross-model verification protocol; sympy verifier (Wave 2 agent_C, mostly failed but partial use).
- **Cost/Value**: 1 week. Value: low-med — narrow but principled.

---

# K. Photography & instrumentation

## K1. Exposure bracketing

- **Source**: Adams, A. (1981), *The Negative*, Little, Brown. Standard practice for high-dynamic-range photography.
- **Practice**: When the correct exposure is uncertain, take 3-7 photos at progressively different settings (e.g., -2EV, -1EV, 0EV, +1EV, +2EV). Pick the best later, or HDR-merge them. The choice of bracket spread is calibrated to scene dynamic range.
- **CoT-CP mapping**: Bracket the *temperature* (or sampling parameter): generate 3 traces at T=0.3, 0.7, 1.0. Different from K=4 majority because the temperatures are *systematically* spread, not all the same. Hypothesis: high-T sample explores creative solutions; low-T sample is rigorous; mid-T is the consensus. Combining all three with temperature-aware weighting gives a Pareto-balanced answer.
- **Concrete proposal**: T-bracket SC: 3 samples at T=0.3, 0.7, 1.0, weight by inverse-temperature when voting. Compare to SC@3 at T=0.7. Likely small gain but cheap.
- **Connects to**: SC family extension; sampling strategy.
- **Cost/Value**: 3 days. Value: low-med.

## K2. Calibration cards / gray cards

- **Source**: ANSI standard photometric reference (Munsell N5 = 18% gray). Standard photography practice since c. 1940.
- **Practice**: Include a reference object of known properties in every shot — a gray card for white balance, a color checker, a ruler for scale. Allows post-hoc correction.
- **CoT-CP mapping**: Include a *reference problem* in the prompt — a known-easy version of the target problem with the answer given. The model's solution to the reference is the calibration; if the model gets the reference wrong, abstain on the target. Provides a per-prompt validity check that's free (one extra easy problem in the prompt).
- **Concrete proposal**: For AIME problems, prepend a known-solved IMO-1985 problem with given answer. The model's reproduction of the answer becomes a *calibration witness* — if it can't reproduce, the prompt is broken (or model is hallucinating); abstain. Cheap diagnostic.
- **Connects to**: Prompt engineering; abstention triggers.
- **Cost/Value**: 1 week. Value: med — adds a sanity check.

---

# L. Forensics & investigation

## L1. Chain of custody

- **Source**: Federal Rules of Evidence Rule 901 (2024). Forensic standard since c. 1900.
- **Practice**: Every piece of evidence from collection to courtroom must have an unbroken, documented chain of custody — who handled it, when, where, for what purpose. Any break voids the evidence's admissibility.
- **CoT-CP mapping**: Every numerical value in a CoT trace should be traceable: *where did this number come from?* (a) given in the problem, (b) computed at step N, (c) recalled from training (a constant like π). Today's traces blur these. Chain-of-custody-style traces would tag each value with its provenance, enabling a downstream verifier to spot-check that "the 7" in step 12 actually came from "the 7" in step 5 (and not a hallucination).
- **Concrete proposal**: Output schema requiring `[value, source]` pairs. e.g., "*Step 5: 7 (given). Step 12: 7² = 49 (computed from step 5).*" Verifier can mechanically check each Step-N derivation against its claimed sources. Beats free-form prose for spot-checking.
- **Connects to**: Wave 2 agent_C (sympy verifier); structured CoT.
- **Cost/Value**: 2 weeks (prompt + parser). Value: med-high — enables mechanical verification.

## L2. Cognitive interview (PEACE model)

- **Source**: Fisher, R. P. & Geiselman, R. E. (1992), *Memory-Enhancing Techniques for Investigative Interviewing*, Charles C. Thomas. UK police PEACE model (1993).
- **Practice**: Witness-recall protocol. Phases: *Plan*, *Engage*, *Account*, *Closure*, *Evaluate*. Account uses techniques: free recall (no leading), context reinstatement (mental return to scene), reverse order, change perspective. Reduces witness confabulation by 30-50% vs naive interview.
- **CoT-CP mapping**: When a CoT trace stalls or seems wrong, today's recourse is regenerate from scratch or backtrack. Cognitive-interview-style would: *change the perspective* on the problem (re-state from a different viewpoint), *reverse-order* (work backward from the desired answer), *context-reinstate* (re-read the original problem). These prompts are different from "let me reconsider" — they're *structured* recall enhancement.
- **Concrete proposal**: When a step's confidence is low, inject one of: (i) *"Re-state the problem in your own words.";* (ii) *"Work backward from the answer you expect.";* (iii) *"Solve a simpler version first.";* and continue. Compare to single rewrite-cue (Pilot L, +1pp). Hypothesize the *structured* alternatives outperform a generic "reconsider" cue.
- **Connects to**: Pilot L (rewrite-cue); Layer B branching.
- **Cost/Value**: 1 week. Value: med.

---

# M. Cooking & operational practice

## M1. Mise en place

- **Source**: Ruhlman, M. (2007), *The Elements of Cooking*, Scribner. Escoffier (1903) brigade system.
- **Practice**: Before starting to cook, all ingredients are measured, prepped, and arranged in order of use. The actual cooking is then a deterministic execution. The *separation* of preparation and execution is what enables high-throughput restaurant kitchens.
- **CoT-CP mapping**: Today's CoT mixes problem-parsing, formula-recall, and computation in one stream. Mise-en-place would split into two phases: (a) *prep* — list all given values, identify what's asked, recall relevant formulas, identify edge cases; (b) *cook* — execute the calculation. The benefit: prep can be scored independently from cook (different score functions per phase). Forces the model to commit to a plan before computing.
- **Concrete proposal**: 2-phase prompt: *"Phase 1 (Prep): list givens, identify unknown, recall relevant formulas, list edge cases. Phase 2 (Solve): execute. Each phase is scored independently."* Allows phase-conditional CP — if Prep is good but Solve fails, abstain; if Prep fails, regenerate Prep before Solve.
- **Connects to**: IRAC (D1); structured CoT; phase-conditional CP.
- **Cost/Value**: 1 week. Value: med — sister to IRAC.

## M2. Tasting at intervals

- **Source**: Standard culinary practice. James Beard (1972), *Beard on Food*.
- **Practice**: Taste the dish 3-4 times during cooking, adjust seasoning each time. Don't taste only at the end (too late to fix). The taste-test is a *verification with low stakes* — if salt is high, you can dilute; if it's low, you add more.
- **CoT-CP mapping**: Periodic mid-trace verification — at fixed token intervals, ask the model "is the current solution path on track?" The intervention is *cheaper* than full per-step PRM scoring and *earlier* than end-of-trace evaluation. Sweet spot between Approach A (per-step) and end-of-trace CP.
- **Concrete proposal**: Mid-trace probe at 25%, 50%, 75% of expected length: *"Pause. So far, are you on track? Brief answer: [on track / off track / unsure]."* If "off track" or "unsure" with high confidence, regenerate from that point. Compare to per-step (expensive) and end-of-trace (too-late).
- **Connects to**: Per-step CP (sparser variant); long-CoT model handling.
- **Cost/Value**: 1 week. Value: med — reasonable middle ground.

---

# Top-12 actionable picks (rank-ordered for v1 paper or near-term v2)

1. **A4. Morbidity & Mortality conferences** (HIGH value, 2-3 days). Builds a failure-mode taxonomy. Per-mode kept-accuracy is a strong §5 figure.
2. **A1. DDx prompting** (HIGH value, 2 days). Forces alternative-hypothesis enumeration. Direct attack on premature-commitment failure.
3. **B1. Pre-decode checklist** (HIGH value, 1 week). Forced step-coverage before solving. Borrows Gawande's strongest result.
4. **D1. IRAC structured prompting** (HIGH value, 1 week). Per-component scoring (Issue / Rule / Application / Conclusion).
5. **F2. After-Action Review automation** (HIGH value, 1 week). Turns failures into improvement signal.
6. **F3. Red Team via cross-model verification** (HIGH value, 1 week). Leverages existing CMV infrastructure.
7. **E2. Five Whys** (HIGH value, already planned). Theoretical justification for HGJ Idea 1.1 (earliest-bad-step re-roll).
8. **L1. Chain of custody output schema** (med-high, 2 weeks). Mechanical step-by-step verifiability.
9. **B2. CRM 2-pilot redundancy** (med-high, 1-2 weeks). Mandatory adversarial verification.
10. **F1. OODA loop prompting** (med-high, 1 week). Forces problem-class identification before solving.
11. **C1. Surgical time-out for critical ops** (med, 1 week). Pre-commit verification at substitution / case-split / limit.
12. **I1. Two-source rule** (med-high, 1-2 weeks). Independent multi-model verification.

---

# Items deferred to v2 / future work

- A2 (blinded concurrence), A3 (SBAR verbalized confidence) — narrow utility
- B3 (sterile cockpit), B4 (NTSB-style archive) — long horizon
- C2 (sponge count), C3 (sterile field), D3 (Daubert framing), D4 (cross-exam) — supplementary or pure framing
- E1 (FMEA workshop) — 1-day exercise, do once
- E3 (SPC charts), E4 (DMAIC), E5 (Poka-yoke) — meta-methodology
- G1 (sectional rehearsal), G2 (jazz fakebook) — narrow
- H1, H2, H3 (chess/sport) — interpretation/aesthetic
- I2 (inverted pyramid), I3 (editorial chain) — engineering
- J1 (BATNA), J2 (CBMs) — framing
- K1 (T-bracket), K2 (calibration card) — small
- L2 (cognitive interview) — Pilot-L extension
- M1 (mise en place), M2 (tasting at intervals) — sisters to IRAC / per-step CP

---

# How this folds into existing project documents

| Existing doc | Update |
|---|---|
| `theorems/PAPER_OUTLINE.md` | Add §6 "future work" subsection citing A1, B1, D1, F2 as principled alternatives to current heuristic prompting |
| `score_ideation/synthesis/RESULTS_SUMMARY_KR.md` | Add a "Wave 5 brainstorm: cross-domain practice imports" section listing the top-12 |
| `score_ideation/from_agents_step/` | Could spawn 5 new agents per the cross-domain analogies (DDx, Checklist, IRAC, AAR, OODA) — Wave 6 brainstorm |
| `experiments/src/` | Implement A1 (DDx), B1 (checklist), D1 (IRAC), F2 (AAR-judge) as 4 new prompt experiments |
| `paper §6 discussion` | Frame CoT-CP as the *statistical-discipline* analog of these *operational-discipline* practices in other fields |

---

# Bibliographic backbone (the 25 most-citable sources here)

| # | Source |
|---|---|
| 1 | Gawande, A. (2009), *The Checklist Manifesto*, Metropolitan Books |
| 2 | Haynes, A. B. et al. (2009), *NEJM* 360:491-499, WHO Surgical Safety Checklist |
| 3 | Boyd, J. R. (1976-1995), *A Discourse on Winning and Losing*, US Army War College archives |
| 4 | Coram, R. (2002), *Boyd: The Fighter Pilot Who Changed the Art of War*, Little, Brown |
| 5 | US Army (2008), *FM 7-0 Training for Full Spectrum Operations*, AAR doctrine |
| 6 | US Army TRADOC (2010), Pam 525-7-19, *Red Teaming Handbook* |
| 7 | Ohno, T. (1988), *Toyota Production System*, Productivity Press |
| 8 | Shingo, S. (1986), *Zero Quality Control*, Productivity Press |
| 9 | Pyzdek, T. & Keller, P. (2018), *The Six Sigma Handbook*, McGraw-Hill |
| 10 | Shewhart, W. A. (1931), *Economic Control of Quality of Manufactured Product*, Van Nostrand |
| 11 | MIL-P-1629 (1949), *Procedures for FMECA*, US DoD |
| 12 | Bickley, L. S. (2020), *Bates' Guide to Physical Examination*, Wolters Kluwer |
| 13 | Joint Commission (2014), *National Patient Safety Goals* NPSG |
| 14 | Leonard, M. et al. (2004), *BMJ Quality & Safety* — SBAR |
| 15 | Pierluissi, E. et al. (2003), *JAMA* — M&M conferences |
| 16 | Knaus, W. A. et al. (1985), *Crit Care Med* — APACHE-II |
| 17 | Helmreich, R. L. et al. (1999), *Int J Aviation Psych* — CRM |
| 18 | NTSB (1962-present), public Aviation Accident Reports |
| 19 | Fisher, R. & Ury, W. (1981/2011), *Getting to Yes*, Penguin |
| 20 | Kovach, B. & Rosenstiel, T. (2014), *The Elements of Journalism*, Three Rivers |
| 21 | Wigmore on Evidence (1904/1972), §1367 — cross-examination |
| 22 | *Daubert v. Merrell Dow* (1993), 509 U.S. 579 |
| 23 | Fisher, R. P. & Geiselman, R. E. (1992), *Memory-Enhancing Techniques* — PEACE/cognitive interview |
| 24 | Levine, M. (1995), *The Jazz Theory Book*, Sher Music |
| 25 | Ruhlman, M. (2007), *The Elements of Cooking*, Scribner |

---

# Companion files

- `CROSS_DISCIPLINARY_IDEAS.md` — the AI/stat/ML-adjacent cross-disc ideas (e-process, BOCPD, BH-FDR, etc.). Kept as-is; this file is the non-AI counterpart.
- `papers/ANALYSIS.md` — 30-paper deep synthesis (within-AI literature)
- `papers/RELATED_WORK_DRAFT.md` — §2 prose for paper

---

## Source URLs

- [Gawande, *Checklist Manifesto* — atulgawande.com](https://atulgawande.com/book/the-checklist-manifesto/)
- [WHO Surgical Safety Checklist — Haynes 2009 PubMed](https://pmc.ncbi.nlm.nih.gov/articles/PMC2647491/)
- [OODA loop — Wikipedia](https://en.wikipedia.org/wiki/OODA_loop)
- [Army War College on Boyd & OODA](https://warroom.armywarcollege.edu/podcasts/boyd-ooda-loop-rr/)
- [FMEA — Wikipedia](https://en.wikipedia.org/wiki/Failure_mode_and_effects_analysis)
- [ASQ on FMEA](https://asq.org/quality-resources/fmea)
- [Five Whys — Adobe blog](https://business.adobe.com/blog/basics/5-whys-root-cause-analysis)
- [IRAC — Wikipedia](https://en.wikipedia.org/wiki/IRAC)
- [ABA Student Lawyer on IRAC](https://www.americanbar.org/groups/law_students/resources/student-lawyer/student-essentials/legal-reasoning-its-all-about-irac/)
- [Columbia Law on IRAC](https://www.law.columbia.edu/sites/default/files/2021-07/organizing_a_legal_discussion.pdf)
