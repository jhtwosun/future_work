# Mathematics, Pure Science & Engineering Practices for CoT

> Compiled 2026-05-08. Third companion to `CROSS_DISCIPLINARY_IDEAS.md` (stat/ML-adjacent)
> and `NON_AI_DOMAIN_PRACTICES.md` (medicine/aviation/law/military). This file covers the
> middle ground: practices from **pure mathematics, physics, chemistry, astronomy, civil/
> mechanical/electrical/chemical engineering** — domains where careful reasoning has been
> codified for centuries and where the CoT analog is concrete.

---

## Why this file

Mathematics, physics, chemistry, and engineering have spent 200-400 years developing
*sanity checks, verification protocols, and structured reasoning conventions* that LLMs
mostly ignore. A typical AIME-level CoT trace skips dimensional analysis, omits special-case
checks, and rarely references conservation laws — even though every working mathematician
would do these reflexively. This catalog imports the codified practices and proposes
concrete CoT-prompting / scoring / evaluation hooks for each.

**Selection rule**: each entry must be a *named, codified* practice from a hard-science /
engineering discipline with a non-trivial mapping to CoT design. Trivial mappings ("just
do dimensional analysis") get rejected; what we want is a specific *protocol* and a
specific *experimental hook*.

---

# A. Pure mathematics

## A1. Pólya's *How to Solve It* — 4-step heuristic

- **Source**: Pólya, G. (1945/2014), *How to Solve It: A New Aspect of Mathematical Method*, Princeton University Press. >1M copies sold across 21 languages. See [Wikipedia](https://en.wikipedia.org/wiki/How_to_Solve_It) and [LibreTexts coverage](https://math.libretexts.org/Courses/Coalinga_College/Math_for_Educators_(MATH_010A_and_010B_CID120)/05:_Problem_Solving/5.02:_George_Polya's_Strategy).
- **Practice**: Four mandatory phases. (1) **Understand**: Identify unknown / data / condition. Restate in your own words. (2) **Plan**: Find connection between data and unknown. Have you seen a related problem? Specialize? Generalize? Vary the problem? (3) **Carry out**: Execute the plan, checking each step. (4) **Look back**: Verify the result. Can you derive it differently? Use it for other problems? The *Look Back* step is where most students fail; Pólya emphasizes it is non-optional.
- **CoT-CP mapping**: Almost no LLM CoT does the Look Back step. Today's "Let me verify..." is rare and unstructured. Pólya's specific Look Back questions ("can you check the result?", "can you derive it differently?", "does it satisfy the original conditions?") are concrete and prompt-engineerable. Different from CoVe (Chain-of-Verification, Dhuliawala+ 2024) which generates verification questions ad hoc — Pólya's are *prescribed*.
- **Concrete proposal**: System prompt: *"After producing your answer, perform Pólya's Look Back: (i) substitute your answer back into the original problem and verify it satisfies the conditions; (ii) check special cases (zero, identity, boundary); (iii) confirm units / orders of magnitude. Output `[VERIFIED]` if all three pass, else `[FAILED: <reason>]`."* Use the `[VERIFIED]` flag as a binary score `s_t = LookBackPassed ∈ {0, 1}`. Compare to baseline kept-accuracy.
- **Connects to**: Wave 5 prompt engineering; verification primitive; close cousin of WHO Surgical Time-out (NON_AI §C1).
- **Cost/Value**: 2-3 days. Value: **HIGH** — the verification is concrete and the score is a binary signal usable in CRC.

## A2. Lakatos *Proofs and Refutations* — proofs as dialectic

- **Source**: Lakatos, I. (1976), *Proofs and Refutations: The Logic of Mathematical Discovery*, Cambridge University Press. Posthumous; based on his 1959 PhD thesis.
- **Practice**: Mathematical proof is not monotone. A proposed proof generates *counterexamples*, which split into "monsters" (rejected as outside the implicit domain) vs "real" counterexamples (which force conjecture revision). The dialectic — conjecture, refutation, monster-barring, lemma-incorporation — is how proofs become rigorous. Key insight: the *first* version of a proof is almost always wrong in subtle ways; rigor is reached through adversarial refinement.
- **CoT-CP mapping**: Today's CoT produces a single proof attempt. Lakatos-style would: (1) generate a proof; (2) generate *counterexamples* to the proof; (3) classify each as monster (boundary case to exclude) or real (forces revision); (4) revise. This is the formal version of CRM (NON_AI §B2) for math reasoning. Distinct from cross-examination (NON_AI §D4) because Lakatos's process is *cooperative* — both parties want the truth, not a win.
- **Concrete proposal**: 3-pass proof: (1) generate proof; (2) "List 3 boundary or special cases that might break this proof"; (3) "For each, determine: monster (excluded by problem statement) or real counterexample (proof must be revised). If real, revise."* Score traces by whether the dialectic surfaced any real counterexamples and whether the revision succeeded. Especially valuable for Olympiad geometry / combinatorics where edge cases dominate.
- **Connects to**: NON_AI §D4 (cross-examination); §B2 CRM. New verification primitive.
- **Cost/Value**: 1 week. Value: med-high — most concrete value on Olympiad-level problems.

## A3. Erdős's "Book proofs" — Pareto over multiple proofs

- **Source**: Aigner, M. & Ziegler, G. M. (2018), *Proofs from THE BOOK*, 6th ed., Springer. Inspired by Erdős's notion of "The Book" — God's collection of perfect proofs.
- **Practice**: For important theorems, mathematicians collect *multiple proofs* (algebraic, combinatorial, geometric, probabilistic) and rank them by elegance / informativeness / generality. The "best" proof depends on use: combinatorial proofs reveal structure; algebraic proofs are short; probabilistic proofs scale. The cardinality-of-proofs is itself a measure of a theorem's importance.
- **CoT-CP mapping**: Self-consistency at N=8 generates 8 possibly-different proofs and majority-votes the answer. Erdős would say: don't vote — *rank* them by independent criteria. The model can compute multiple scores per trace (lp_min, prm_min, sc_top1, our `tiebreak_lex`); a Book-proof aggregator would pick the trace with best *combined* score, not majority answer. Different from majority because (a) it never picks an unconvincing trace just because most agree, (b) it incorporates trace quality, not just answer agreement.
- **Concrete proposal**: For each (prompt, model) generate K=4 traces with diverse temperatures; score each by `tiebreak_lex` quality; pick the highest-quality. Compare to majority-vote. Hypothesis: for hard problems where most traces are wrong, picking by quality rather than vote dominates. Tested on AIME / Olympiad.
- **Connects to**: Layer A `tiebreak_lex` use; multi-trace selection.
- **Cost/Value**: 3-5 days. Value: med — likely small absolute lift over `tiebreak_lex`.

## A4. Tao's blog — explicit metacognitive principles

- **Source**: Tao, T., *What's New* blog (2007-present), terrytao.wordpress.com. Particularly [the "advice on..." career pages](https://terrytao.wordpress.com/career-advice/) and [solving mathematical problems](https://terrytao.wordpress.com/career-advice/solving-mathematical-problems/).
- **Practice**: Tao distinguishes "rigorous", "post-rigorous", and "pre-rigorous" stages of mathematical thought. Pre-rigorous: heuristic, approximate. Rigorous: formal, careful, slow. Post-rigorous: heuristic again, but informed by absorbed rigor. Most students are stuck in rigorous; many LLMs are stuck in pre-rigorous. The *insight* is that fluent math reasoning *alternates* between heuristic and rigorous modes deliberately.
- **CoT-CP mapping**: A CoT that is uniformly "rigorous-mode" is verbose and slow; uniformly "pre-rigorous" is fast but unreliable. Tao-style would *explicitly mark* each step as heuristic / rigorous: *"[heuristic] The answer is around 70."* vs *"[rigorous] By Cauchy-Schwarz, ..."*. Heuristic steps are sanity checks (cheap, give Fermi estimates); rigorous steps are derivations. Each gets a different score function.
- **Concrete proposal**: Two-mode CoT: heuristic (short, no proof, output is a Fermi estimate) followed by rigorous (formal derivation). Final answer must lie within ±20% of the heuristic estimate; mismatch flags the trace for re-examination. Implements Tao's "post-rigorous" check on top of standard CoT.
- **Connects to**: Sanity check primitives; cf. Dimensional Analysis (B1), Fermi estimation (B2).
- **Cost/Value**: 1 week. Value: med-high — the heuristic-vs-rigorous mismatch is a cheap and informative signal.

## A5. Reverse mathematics (Friedman, Simpson)

- **Source**: Simpson, S. G. (2009), *Subsystems of Second Order Arithmetic*, 2nd ed., Cambridge University Press. Friedman's program from c. 1975.
- **Practice**: For each theorem, identify the *minimal* axiom system needed to prove it. Most "natural" theorems use one of five base systems (RCA₀, WKL₀, ACA₀, ATR₀, Π¹₁-CA₀). Reverse mathematics proves *equivalences*: e.g., Bolzano-Weierstrass is equivalent over RCA₀ to König's Lemma. The discipline shifts from "what can we prove?" to "what assumptions does this proof use?".
- **CoT-CP mapping**: For an LLM proof, *which assumptions does the model invoke?* If the model uses an unjustified theorem ("by AM-GM" without checking AM-GM applicability), that's a hidden axiom. A reverse-math-style score would extract the named theorems / lemmas / facts the trace uses and verify each is (a) real, (b) correctly stated, (c) correctly applicable. This gives a per-step "axiom legitimacy" score complementary to PRM.
- **Concrete proposal**: Post-process traces to extract every named-theorem invocation (regex on "by [Theorem Name]"); query a knowledge-base (Wikipedia API or formal libraries like mathlib4) to verify the theorem exists and the citation is correct. Per-trace score = fraction of valid citations. Different from PRM because it checks *factual* correctness, not just step plausibility.
- **Connects to**: New score family — citation-validity score; complements PRM.
- **Cost/Value**: 2 weeks (KB integration). Value: med — narrow but very interpretable.

## A6. Formal proof assistants (Lean 4, Coq, Isabelle)

- **Source**: de Moura, L. et al. (2021), *Lean 4: A Theorem Prover and Programming Language*, CADE 28. Mathlib4 community.
- **Practice**: Express proof in a dependent type theory; the type-checker verifies correctness mechanically. Recent achievements: Carleson's theorem formalized (2024, 100K lines Lean); IMO-2024 P6 by AlphaProof (2024, DeepMind, silver-medal level); BlueOrange formalization of LLM-generated proofs (2025).
- **CoT-CP mapping**: A Lean-verifiable proof gives binary ground truth on correctness — orthogonal to PRM (which scores plausibility). For domains where Lean can express the problem (algebra, analysis, simple combinatorics), routing CoT through a proof-assistant gives perfect verification. The 2024 AlphaProof system already does this for IMO geometry/algebra. The CoT-CP analog: use Lean as a high-cost verifier in the score-family ladder, sitting above `sc_top1` (since Lean proofs cost more than 8× a sample).
- **Concrete proposal**: For a subset of MATH-500 / OlympiadBench problems with autoformalizable statements, run Lean tactics (or call Lean-trained model like Llemma) on the LLM-produced proof; accept iff Lean accepts. Cost: 30-60 sec per Lean attempt. Cost/value depends on autoformalization success rate. Currently DeepMind autoformalization is ~30% on Olympiad problems.
- **Connects to**: Top of the score-family ladder; Wave 2 agent_C external verifier (sympy is the cheap analog).
- **Cost/Value**: 2-3 weeks. Value: high if autoformalization rate exceeds 25% — gives ground-truth verification on a meaningful subset.

---

# B. Physics

## B1. Buckingham π theorem & dimensional analysis

- **Source**: Buckingham, E. (1914), *On physically similar systems; illustrations of the use of dimensional equations*, Physical Review 4 (4): 345-376. See [Wikipedia](https://en.wikipedia.org/wiki/Buckingham_pi_theorem) and [MIT 2.25 lecture notes](https://ocw.mit.edu/courses/2-25-advanced-fluid-mechanics-fall-2013/c0a4521f55e9191d557c167e99e97469_MIT2_25F13_The_Buckingham.pdf). Standard in physics / fluid dynamics curriculum since 1920s.
- **Practice**: Given n physical variables involving k base dimensions (e.g., [length], [time], [mass]), the number of independent dimensionless groups is n − k. Every physical equation can be rewritten as a relationship between dimensionless groups (π's). The practice gives: (a) a free sanity check — the proposed equation must be dimensionally homogeneous; (b) a parsimony argument — the answer can depend on at most n − k independent quantities.
- **CoT-CP mapping**: Math problems with dimensional content (physics word problems, geometry with units, applied math) admit a free verification: the answer's units must match the question's expected units. Today's CoT mostly drops units mid-trace. A dimensional-check primitive would parse units from the prompt, propagate them through the trace, and flag inconsistency. Different from sympy verification (Wave 2 agent_C, mostly failed) because dimensional analysis succeeds on text without algebraic manipulation.
- **Concrete proposal**: Implement a unit-tracking parser; for any prompt with explicit units, validate the final answer's units match. For pure-math problems without units, the parser is a no-op. Use the binary "units valid" as a step in `prm_min` aggregation. Should give a small but free lift on physics-flavored AIME problems.
- **Connects to**: Sanity-check primitive; complements sympy verifier.
- **Cost/Value**: 1 week. Value: med — narrow but free.

## B2. Fermi estimation / order-of-magnitude bounds

- **Source**: Weinstein, L. & Adam, J. A. (2008), *Guesstimation: Solving the World's Problems on the Back of a Cocktail Napkin*, Princeton. Original: Enrico Fermi's 1945 atomic-bomb-yield estimate from displaced confetti.
- **Practice**: Before solving a quantitative problem rigorously, produce a rough estimate by chained multiplication of order-of-magnitude estimates. Each estimate has known error (factor of 3 to 10); combined error grows multiplicatively but slowly enough that the final estimate is usually within a factor of 10. Used as a *sanity check* on rigorous calculation.
- **CoT-CP mapping**: A free *consistency check* between the rigorous answer and a Fermi estimate. If the rigorous CoT gives answer 47 and a Fermi estimate gives "around 50," consistency is high. If the CoT gives 4700 and Fermi gives 50, the CoT is suspect. Implements Tao's heuristic-rigorous duality (A4) as a concrete numerical check.
- **Concrete proposal**: Two-pass CoT: pass 1 generates a Fermi estimate ("approximately 5000, give or take 10×"); pass 2 generates the rigorous answer. Accept iff the two agree within the Fermi pass's stated bounds. Hypothesis: catches gross arithmetic errors (off-by-10× from a misplaced decimal) very cheaply.
- **Connects to**: A4 Tao heuristic-rigorous; A1 Pólya Look Back.
- **Cost/Value**: 3-5 days. Value: med-high — cheapest sanity check available, especially valuable for AIME where answers are 0-999 integers.

## B3. Conservation laws as sanity checks

- **Source**: Noether, E. (1918), *Invariante Variationsprobleme*, Nachrichten Göttingen. See Goldstein, Poole, Safko (2002), *Classical Mechanics*, 3rd ed.
- **Practice**: Energy, momentum, charge, baryon number, lepton number, parity, etc. Each is conserved in physical processes. After deriving a result, verify the conservation laws hold. Failure means a sign error, a missing term, or a wrong identification of the process.
- **CoT-CP mapping**: For combinatorial / counting problems, the conservation law is "everything is counted exactly once." For probability problems, "probabilities sum to 1." For geometry, "angles in a triangle sum to π." These domain-specific conservation checks are free; LLMs frequently violate them. A conservation-check primitive would identify the active conservation law and verify it on the answer.
- **Concrete proposal**: Domain-specific check library: combinatorics (∑ subset sizes = total), probability (∑ pᵢ = 1), geometry (∑ angles), arithmetic series (sum formula). Apply automatically when a relevant pattern is detected. Free, mechanical, gives a binary score.
- **Connects to**: B1 dimensional analysis (sister sanity check); A1 Pólya Look Back step (i).
- **Cost/Value**: 1 week (build library). Value: med — narrow each, but combined gives several free checks per problem.

## B4. Limiting / boundary case verification

- **Source**: Standard in mathematical physics. Boas, M. L. (2005), *Mathematical Methods in the Physical Sciences*, 3rd ed., Wiley, ch. 2.
- **Practice**: Test a derived formula at limiting values (zero, infinity, identity, special angles like 0, π/4, π/2). The formula must reduce to a known correct expression in each limit. Catches sign errors, off-by-one errors, and missing factors.
- **CoT-CP mapping**: Almost no LLM trace tests its derived expression at limits. Pólya's Look Back step (i) "check special cases" is exactly this. As a primitive: identify the free variables in the derived expression; substitute extreme values (0, 1, ∞); verify the result is sensible in each limit.
- **Concrete proposal**: For symbolic-answer problems (AIME problems with parameter), automatic limit-check: identify variable, substitute limits, verify. Could combine with sympy for the symbolic substitution. For numerical-answer problems, less applicable but still useful for verifying the general approach.
- **Connects to**: A1 Pólya Look Back; A4 Tao heuristic-rigorous.
- **Cost/Value**: 1-2 weeks. Value: med-high for parameter-flavored Olympiad problems.

## B5. 5σ discovery threshold (particle physics)

- **Source**: CERN (2012), Higgs discovery announcement; see [CERN Courier "Five sigma revisited"](https://cerncourier.com/a/five-sigma-revisited/) and [CERN FAQ on 5σ](https://home.cern/resources/faqs/five-sigma). Adopted from Feynman / 1960s SLAC tradition.
- **Practice**: A particle-physics discovery requires a local p-value ≤ 3 × 10⁻⁷ (5σ). The reason is not pure statistics — it's a calibrated response to *look-elsewhere effects* (multiple comparisons), *systematic uncertainty underestimation*, and *prior odds of new physics being false*. The 5σ threshold encodes "we've been burned before by 3σ flukes."
- **CoT-CP mapping**: Our paper claims P(correct) ≥ 1−α with bootstrap CIs. The 5σ tradition argues we should over-reserve to account for: (a) our own multiple-testing — we report 5+ α values × 11 models × 7 datasets = 385+ comparisons; (b) systematic uncertainty in PRM training; (c) the prior probability our method generalizes. Particle-physics-style framing would reserve ~10× our nominal slack to account for non-statistical sources of error. Pragmatically: report results with α-margin (e.g., target α=0.10, achieve α=0.05 empirically) as the headline.
- **Connects to**: §5 reporting discipline; bootstrap CI computation (already done with 500 boot × 10 splits).
- **Cost/Value**: 0 (reporting decision). Value: low-med — sharpens framing.

## B6. Blind analysis (HEP, cosmology)

- **Source**: Klein, J. R. & Roodman, A. (2005), *Blind analysis in nuclear and particle physics*, Annual Review of Nuclear and Particle Science 55: 141-163.
- **Practice**: All analysis decisions (cuts, fitting procedures, systematic-error estimates) are made *before* unblinding the result. After unblinding, no changes are allowed except documented bug fixes. Prevents conscious or unconscious tuning to a desired result. Standard in HEP since c. 2000; cosmology adopted post-2010.
- **CoT-CP mapping**: We currently develop scores by looking at MATH-500 results, then run AIME. This is sequential — each new dataset is partially unblinded by what we learned from previous ones. A blind protocol would: pre-register the score-family ladder, the calibration procedure, and the evaluation metric *before* looking at AIME / OlympiadBench / GPQA. Then run all benchmarks with no further tuning.
- **Concrete proposal**: For the v1 paper, document a "blind run protocol" as an appendix: state which scores, α-grid, and evaluation pipeline were fixed in advance; report which (if any) decisions were made post-unblinding. Adds reviewer-credibility cheaply. Particularly powerful for the OOD section (Theorem 3 evaluation on AIME) where Goodhart's law is a real risk.
- **Connects to**: Reproducibility checklist; honest reporting in §5.
- **Cost/Value**: 0 (documentation). Value: med — high reviewer-credibility per cost.

---

# C. Chemistry

## C1. Retrosynthesis (Corey)

- **Source**: Corey, E. J. (1967), *General methods for the construction of complex molecules*, Pure & Applied Chemistry 14: 19. Nobel Prize 1990 for retrosynthetic analysis.
- **Practice**: To synthesize a complex target molecule, work *backward* from the target to known starting materials by repeatedly applying transforms (functional group interconversions, disconnections at strategic bonds). Each backward step must correspond to a forward synthesis. The disconnection tree explores multiple routes; the chemist picks the best.
- **CoT-CP mapping**: For math problems, the analog is *backward chaining* — start from the desired answer's form, work back to the given quantities. Combined with forward chaining, gives a meet-in-the-middle proof structure that often closes the gap faster than pure forward search. Used in formal proof assistants (Lean's `omega` tactic, Coq's `auto`); not standard in CoT prompting.
- **Concrete proposal**: Two-direction CoT: forward pass from given-to-conclusion, backward pass from conclusion-form-to-given. Accept iff the two passes meet at a shared intermediate. Specifically valuable for problems where the answer form is constrained (AIME's 0-999 integers, problems with explicit asked-for form).
- **Connects to**: Cognitive interview reverse-order recall (NON_AI §L2); Pólya's "Have you used all the data?" check.
- **Cost/Value**: 1-2 weeks. Value: med — established as effective in formal-proof literature; under-tested in CoT.

## C2. Multi-spectroscopy verification (NMR + IR + MS)

- **Source**: Silverstein, R. M. et al. (2014), *Spectrometric Identification of Organic Compounds*, 8th ed., Wiley. Standard in chemistry curricula.
- **Practice**: To identify an unknown compound, run *multiple* independent spectroscopies (¹H-NMR, ¹³C-NMR, IR, mass spec, sometimes UV-vis or X-ray). Each gives partial information; the structure is uniquely determined only by combining all. *Independence* of methods is the key — they probe different physical properties.
- **CoT-CP mapping**: Direct analog of the journalism two-source rule (NON_AI §I1) but with the principle that *different score functions probe different aspects*. Our `lp_min` (likelihood), `prm_min` (process reward), `sc_top1` (vote-share) are three quasi-independent spectra of trace quality. The Silverstein lesson is: *combine* them, don't pick one. Closer to `tiebreak_lex` than to picking a single best score.
- **Concrete proposal**: Chemistry-style identification table: rows = traces, columns = `lp_min`, `prm_min`, `sc_top1`, `entropy_mean`, `arith_violations`. Accept a trace iff it passes 4 of 5 columns at score-specific thresholds. The "passes 4 of 5" structure is more robust than majority vote because it requires independent confirmation across very different score families.
- **Connects to**: `tiebreak_lex` extension; multi-score combination; cf. NON_AI §C3 sterile field layered redundancy.
- **Cost/Value**: 3-5 days. Value: med — likely small absolute lift but principled.

## C3. Stoichiometry as conservation accounting

- **Source**: Standard in introductory chemistry, e.g., Brown, T. L. et al. (2017), *Chemistry: The Central Science*, 14th ed., Pearson, ch. 3.
- **Practice**: Every chemical equation must *balance* — atoms of each element are conserved; charge is conserved; mass is conserved. Balancing a reaction is mechanical (linear algebra over reaction stoichiometry) but enforces the conservation discipline.
- **CoT-CP mapping**: For combinatorial / counting problems, the analog is "every element is counted once." For algebra problems with many variables, "every variable's role is accounted for." This is a stricter version of conservation laws (B3) — instead of just "the total is conserved," we require "every constituent is tracked." Connected to Chain of Custody (NON_AI §L1).
- **Concrete proposal**: Variable-tracking schema in CoT prompt — at each step, the model must declare which variables are introduced, used, and eliminated. Trace fails if a variable is introduced but never used or eliminated.
- **Connects to**: NON_AI §L1 chain of custody; B3 conservation laws.
- **Cost/Value**: 2-3 days. Value: low-med.

---

# D. Astronomy

## D1. Distance ladder (parallax → Cepheids → SNe Ia → CMB)

- **Source**: Riess, A. G. et al. (2022), *A Comprehensive Measurement of the Local Value of the Hubble Constant*, ApJL 934: L7. Modern review of the cosmological distance ladder.
- **Practice**: Distances to far galaxies cannot be measured directly. Instead, a *ladder* of overlapping methods: parallax (to ~1 kpc), Cepheid variables (to ~30 Mpc), Type Ia supernovae (to several Gpc), CMB (cosmic). Each rung is calibrated against the rung below. Errors compound but are bounded by the calibration overlap.
- **CoT-CP mapping**: Calibration via PRM800K to a ladder of test sets. PRM800K is "rung 1" (in-distribution); MATH-500 is rung 2; Olympiad is rung 3; AIME is rung 4. We currently calibrate at rung 1 and evaluate at rungs 2-4 — assuming the rungs are exchangeable, which they are not (Theorem 3 acknowledges this for AIME). Distance-ladder thinking would: at each rung, *partially recalibrate* using a few labeled examples to handle the distributional drift. This is closer to per-domain calibration than to one-shot.
- **Concrete proposal**: Ladder calibration: PRM800K (n=12K) for vanilla quantile; MATH-500 train split (~250) for first-rung correction; AIME-old years (1983-2014, n=750) for second-rung correction; AIME-2024/25 for evaluation. Each rung corrects the previous via a small affine transformation of the quantile. Theorem 3's empirical-PMF correction is a special case.
- **Connects to**: Theorem 3 OOD calibration; multi-domain transfer.
- **Cost/Value**: 2 weeks. Value: med — natural extension of Theorem 3; could become a §5.7 figure.

## D2. Multi-wavelength survey design

- **Source**: Standard in astronomical survey design; e.g., LSST/Vera Rubin Observatory science requirements (2017).
- **Practice**: A multi-wavelength survey observes the same patch of sky in multiple bands (UV / optical / IR / radio). Different bands probe different physics (UV: hot stars; IR: dust; radio: synchrotron). A source detected only in one band is suspect; a source detected across bands is confirmed.
- **CoT-CP mapping**: Cross-confirmation across score families (cf. C2 multi-spectroscopy). A trace whose answer is supported by *multiple* score families (high lp_min AND high prm_min AND high sc_top1) is more likely correct than one supported by only one. Closely related to multi-spectroscopy (C2) — same family of practice across hard sciences.
- **Connects to**: C2 multi-spectroscopy; `tiebreak_lex`.
- **Cost/Value**: shared with C2.

---

# E. Civil & mechanical engineering

## E1. Factor of Safety (FoS)

- **Source**: ASCE 7-22 (2022), *Minimum Design Loads and Associated Criteria for Buildings and Other Structures*. AISC Steel Construction Manual (15th ed., 2017).
- **Practice**: For structural design, the design load is multiplied by a *factor of safety* (typically 1.5 to 4) before being compared to the structural capacity. The FoS encodes worst-case loads, material variability, construction tolerances, and uncertainty in load distribution. For high-stakes applications (aviation: 1.5; bridges: 2.5-3; nuclear: 4-5+), the FoS is mandated by code.
- **CoT-CP mapping**: We report kept-accuracy at α∈{0.1, 0.3, 0.5}. The Factor of Safety frame says: for stake-X applications, set α = stake-X-allowable-error / FoS. e.g., medical stakes might need α=0.01 to leave a 10× safety factor; casual QA can tolerate α=0.5. Reframes α-grid as a stake-dependent design choice — closer to law's burden-of-proof gradations (NON_AI §D2).
- **Concrete proposal**: §6 deployment guidance: a small table mapping (use case, stake, recommended α, achieved kept-accuracy on our data). Lets a deployment engineer pick α the way a structural engineer picks FoS — by reading off a code table.
- **Connects to**: §6 framing; NON_AI §D2.
- **Cost/Value**: 0. Value: low-med.

## E2. FEA mesh refinement convergence

- **Source**: Cook, R. D. et al. (2002), *Concepts and Applications of Finite Element Analysis*, 4th ed., Wiley, ch. 11.
- **Practice**: For finite-element analysis of stress / heat / flow, refine the mesh until the solution stops changing significantly. Typical convergence criterion: solution norm changes by < 1% upon halving the element size. Lack of convergence indicates either a singular point (real physics) or a numerical artifact.
- **CoT-CP mapping**: For self-consistency, run with N=2, 4, 8, 16; if the majority answer is *stable* across N, accept; if it changes with N, abstain or escalate. This is structurally different from fixed-N SC because the number of samples adapts to the problem. Closer to ESC (early-stopping consistency) but with a *stability* criterion rather than a *zero-entropy* criterion. The mesh-refinement frame gives the convergence test explicit: same answer at consecutive doublings of N.
- **Concrete proposal**: Adaptive-N SC: N=2 → check majority; if stable, accept; else N=4, recheck; ..., up to N=32. Compare to fixed N=16 in compute and accuracy. Hypothesis: matches N=16 accuracy at much lower mean compute.
- **Connects to**: Adaptive-Consistency family (Tier 4 competitor); convergence test.
- **Cost/Value**: 1 week. Value: med — fits naturally into our framework.

## E3. Tolerance stacking

- **Source**: Bjørke, Ø. (1989), *Computer-Aided Tolerancing*, ASME Press.
- **Practice**: When a chain of dimensions is assembled (e.g., 5 parts each 100mm ± 0.1mm), the worst-case stack-up is 5 × 0.1 = 0.5mm; the statistical (RSS) stack-up is √5 × 0.1 ≈ 0.22mm. Designers must decide whether worst-case or RSS analysis applies based on whether errors are correlated.
- **CoT-CP mapping**: Multi-step CoT *accumulates* error step by step. If each step is correct with probability p, naive worst-case gives P(all correct) = pⁿ; RSS-like (independence) gives the same. But if errors are correlated (one wrong step makes the next more likely wrong), the worst-case bound bites. Our trajectory-CP via aggregator φ implicitly assumes errors are aggregable — the φ choice (min, mean, last) determines whether we're worst-casing or averaging.
- **Concrete proposal**: Theoretical remark in §3 connecting our `lp_min` (worst-case stack), `lp_mean` (RSS-style), and `lp_last` (no stack) to tolerance-stacking traditions in mechanical engineering. Clarifies why min vs mean is a *design choice*, not just a hyperparameter.
- **Connects to**: §3 framing; cf. Theorem 2 LR+ Pareto.
- **Cost/Value**: 0. Value: low — pure framing.

## E4. Critical Path Method (CPM) and PERT

- **Source**: Kelley, J. E. & Walker, M. R. (1959), *Critical-path planning and scheduling*, Eastern Joint Computer Conference. DuPont/Remington Rand.
- **Practice**: Decompose a project into tasks with dependencies and durations. Compute the *critical path* — the longest dependency chain — which determines total project duration. Speeding up non-critical tasks doesn't help; only critical-path tasks matter.
- **CoT-CP mapping**: For multi-step proofs, *which steps are on the critical path?* Pilot 2.1 (earliest-bad-step re-roll) and Five Whys (NON_AI §E2) both implicitly target the critical path: the earliest divergence is more critical than later ones. CPM gives the formal framework — build a dependency graph of the trace, identify critical steps, allocate compute to them. Sister practice to FEA refinement (E2): refine where it matters.
- **Concrete proposal**: Dependency-graph extraction from trace (named-variable usage tracks dependencies). Identify the critical path. For step rejection, only re-roll critical-path steps. Combines naturally with chain-of-custody schema (NON_AI §L1).
- **Connects to**: HGJ Idea 1.1 (earliest-bad-step); Layer B step rejection; chain of custody.
- **Cost/Value**: 2 weeks. Value: med-high — formalizes our existing intuition.

## E5. Reliability engineering / Weibull analysis

- **Source**: Weibull, W. (1951), *A statistical distribution function of wide applicability*, J. Applied Mechanics 18: 293-297. MIL-HDBK-217F (US DoD reliability prediction).
- **Practice**: Component lifetimes follow a Weibull distribution. Reliability engineers fit Weibull parameters to test data and predict mean-time-between-failures (MTBF). Allows budgeting maintenance and warranties.
- **CoT-CP mapping**: A "trace lifetime" = number of correct steps before the first error. Fitting Weibull to step-error positions across the corpus could reveal: (a) mean-trace-length-before-first-error; (b) hazard rate as a function of step position (does error rate accelerate after step 5? plateau? decrease?). Has implications for where to place the cutoff in early-abort schemes.
- **Concrete proposal**: Empirical analysis: for each (model, dataset), fit Weibull to first-error position. Use to derive an *optimal* per-step CP threshold schedule (Bonferroni's α/T_max is too conservative if hazard is increasing).
- **Connects to**: Per-step CP Approach B; long-CoT model handling.
- **Cost/Value**: 1 week. Value: med — gives a principled threshold schedule.

---

# F. Electrical engineering & signal processing

## F1. Hamming codes / Error-Correcting Codes

- **Source**: Hamming, R. W. (1950), *Error detecting and error correcting codes*, Bell System Technical Journal 29 (2): 147-160. Standard in coding theory, MacKay (2003) *Information Theory, Inference, and Learning Algorithms*, ch. 1.
- **Practice**: To transmit a k-bit message over a noisy channel, encode into n>k bits with redundancy. A (n, k) code can correct up to ⌊(d−1)/2⌋ errors where d is the minimum Hamming distance. Hamming (7,4) corrects single-bit errors with 3 redundant bits.
- **CoT-CP mapping**: A CoT trace can be designed with built-in redundancy: have the model state intermediate quantities *twice* in different forms (numeric + symbolic, or English + LaTeX). Discrepancies between the two forms flag errors. The redundancy is overhead; the verification is free downstream.
- **Concrete proposal**: Prompt: *"For each numerical result, state both decimal and (if rational) fractional form."* Mismatch between the two flags an error. Cheap to implement; gives single-error correction at zero compute cost beyond the prompt.
- **Connects to**: Sanity check primitives; cf. Tolerance stacking (E3).
- **Cost/Value**: 2-3 days. Value: low-med — narrow but free.

## F2. Lock-in amplifier / synchronous detection

- **Source**: Stanford Research Systems (1999), *About Lock-In Amplifiers*, application note. Originated in WWII-era radar; modern instrument c. 1955.
- **Practice**: To extract a small signal from a noisy background, modulate the signal at a known frequency (the *reference*), detect at the same frequency, and integrate over time. Signal-to-noise improves as √T (T = integration time). Allows detection of signals 10⁹× weaker than the noise floor.
- **CoT-CP mapping**: When a CoT step's score is low, integrate over multiple paraphrases of the question to extract the signal. Different from N-resampling because the modulation (paraphrase) is *known* and the de-modulation (matching answers) extracts the signal coherently. Closely related to two-source / multi-spectroscopy ideas, but with the explicit "modulate at known frequency" structure.
- **Concrete proposal**: Generate K=4 paraphrases of the question; sample 1 trace each; the signal = the answer that appears across paraphrases. Different from blind concurrence (NON_AI §A2) because we *force* paraphrase diversity. Compare to vanilla SC@4.
- **Connects to**: A2 medical concurrence; paraphrase consensus (Wave 1 Pilot M, currently +5.7pp).
- **Cost/Value**: 3-5 days. Value: med — refines existing Pilot M.

## F3. Kalman filter / recursive Bayesian filtering

- **Source**: Kalman, R. E. (1960), *A new approach to linear filtering and prediction problems*, Journal of Basic Engineering 82 (1): 35-45. Standard text: Bar-Shalom, Y. et al. (2001), *Estimation with Applications to Tracking and Navigation*.
- **Practice**: At each time step, maintain a posterior over the state. *Predict* forward via dynamics; *update* using a noisy measurement. Optimal under linear-Gaussian assumptions; extensions (EKF, UKF, particle filter) handle nonlinearity. Foundational for navigation, control, signal extraction.
- **CoT-CP mapping**: Treat "is the trace correct" as a hidden state to track. At each step: predict via the trace's dynamics (P(correct after step t+1 | correct at step t)); update with the step's score s_t. The posterior P(correct | scores 1..t) is a Bayesian-filtered version of our running confidence. More principled than the running min / mean we currently use; gives a per-step probabilistic guarantee.
- **Concrete proposal**: Implement a 1D Kalman-style filter for the running probability of correctness. Calibrate transition / observation noise on PRM800K. Use the filter's posterior probability as the per-step trigger for early abort or branching. This is mathematically the right way to do per-step CP Approach A; current heuristic threshold is a degenerate filter.
- **Connects to**: Per-step CP Approach A; CROSS_DISCIPLINARY §A1 (e-process); cf. POMDP belief states.
- **Cost/Value**: 2 weeks. Value: high — formalizes Approach A on solid foundations.

## F4. ROC curves / Receiver Operating Characteristic

- **Source**: Marcum, J. I. (1947), *A statistical theory of target detection by pulsed radar*, RAND Memorandum RM-754. Modern: Fawcett, T. (2006), *An introduction to ROC analysis*, PRL 27 (8): 861-874.
- **Practice**: Plot true-positive rate vs false-positive rate as the decision threshold varies. The Area Under Curve (AUC) is a threshold-independent classifier quality. Operating points on the curve are chosen by application-specific cost/benefit (radar: high TPR at low FPR for missile defense; medical: low FPR for cancer screening to avoid unnecessary biopsy).
- **CoT-CP mapping**: We report kept-accuracy at α∈{0.1, 0.3, 0.5} which is essentially three points on a ROC-like curve. Reporting the full curve (AUC for the decision "is this trace correct") is more informative and standard in radar / medical literature. Already implicit in our `E5v2_figure1.png` Pareto plot, but framing it as ROC gives reviewers a familiar handle.
- **Concrete proposal**: §5 figure: ROC curve of (kept-fraction, kept-accuracy) for each score family, with AUC labeled. Compare AUCs across `lp_min`, `prm_min`, `sc_top1`. Adds a single-number summary that complements α-grid tables.
- **Connects to**: §5 figures; threshold-independent score comparison.
- **Cost/Value**: 1 day. Value: low-med — better presentation, no new method.

## F5. Phase-locked loop (PLL)

- **Source**: Gardner, F. M. (2005), *Phaselock Techniques*, 3rd ed., Wiley.
- **Practice**: Lock a local oscillator to an incoming signal's frequency by feedback: detect phase error, low-pass filter it, drive the oscillator. Used in clock recovery, FM demodulation, frequency synthesis.
- **CoT-CP mapping**: Lock the model's reasoning *cadence* to the problem's natural step structure. If the model produces too-short steps (overly choppy) or too-long steps (verbose), inject feedback to adjust. Concretely: monitor average step length during decoding; if outside expected range (calibrated on PRM800K), inject correction tokens. Different from temperature scheduling because the controlled variable is *granularity*, not entropy.
- **Connects to**: Step segmentation (cf. EDU-PRM in our notes); long-CoT handling.
- **Cost/Value**: 2 weeks. Value: low-med — narrow but principled.

---

# G. Chemical & process engineering

## G1. HAZOP (Hazard and Operability Study)

- **Source**: ICI (1960s), originator. Imperial Chemical Industries developed for safety of ammonia plants. See [Wikipedia HAZOP](https://en.wikipedia.org/wiki/Hazard_and_operability_study), AIChE CCPS, [SafetyCulture overview](https://safetyculture.com/topics/hazop). Codified in IEC 61882.
- **Practice**: Decompose a process into *nodes*; for each node, apply *guidewords* (NO/NONE, MORE, LESS, AS WELL AS, PART OF, REVERSE, OTHER THAN) to each *parameter* (flow, pressure, temperature, concentration). For each (guideword × parameter) combination, ask: *what could cause this deviation? what are the consequences? what safeguards exist? are they sufficient?* Generates a comprehensive hazard register.
- **CoT-CP mapping**: Apply HAZOP guidewords to a CoT trace: (i) "NO" — what if we omit this step? (ii) "MORE" — what if we have more of this quantity? (iii) "REVERSE" — what if the inequality flips? (iv) "OTHER THAN" — what if the variable means something different? Each guideword × step-content combination generates a candidate counter-example. Different from Lakatos (A2) which is open-ended; HAZOP gives *specific* probe categories. Builds on FMEA (NON_AI §E1) but more *generative* (FMEA enumerates failures from history; HAZOP generates them via guidewords).
- **Concrete proposal**: Post-trace HAZOP: for each step, prompt the model to apply 3-4 guidewords and check for deviations. If any guideword reveals an inconsistency, flag the step. Resource-intensive (3-4× more LLM calls per trace) but gives systematic coverage.
- **Connects to**: NON_AI §E1 FMEA; A2 Lakatos refutations.
- **Cost/Value**: 1-2 weeks. Value: med — most thorough verification protocol available; high cost.

## G2. Layer of Protection Analysis (LOPA)

- **Source**: AIChE CCPS (2001), *Layer of Protection Analysis: Simplified Process Risk Assessment*. ANSI/ISA-84.00.01-2004.
- **Practice**: For each hazard, identify the *layers of protection* (control system, alarm, safety-instrumented system, pressure relief, containment, emergency response). For each layer, estimate failure probability. Required: total residual risk ≤ tolerable threshold. The product of layer failure probabilities determines the protection factor.
- **CoT-CP mapping**: Different score functions are different *layers of protection* against accepting a wrong answer. `lp_min` catches some errors; `prm_min` catches others; `sc_top1` catches yet others. Independence of layers is the key assumption (failure modes are mostly disjoint). LOPA discipline says: *quantify* the residual risk = product of (1 − recall) across layers. Could be a §5 analysis: empirical residual risk after combining 2, 3, 4 layers.
- **Concrete proposal**: §5 ablation: kept-accuracy as we add score-family layers. (i) `lp_min` only. (ii) `lp_min` AND `prm_min`. (iii) `lp_min` AND `prm_min` AND `sc_top1`. Plot as an "accuracy floor" curve. The flatness of the curve indicates layer dependence (layers catch the same failures); steepness indicates independence (layers catch different failures). Diagnostic for which scores are complementary.
- **Connects to**: NON_AI §C3 sterile field layered redundancy; multi-spectroscopy (C2); E2 mesh refinement.
- **Cost/Value**: 3-5 days. Value: med — clean analysis figure for §5.

## G3. Mass / energy balance

- **Source**: Felder, R. M. & Rousseau, R. W. (2005), *Elementary Principles of Chemical Processes*, 3rd ed., Wiley, ch. 4-5.
- **Practice**: For any process unit, apply conservation of mass (input = output + accumulation) and energy. Discrepancies indicate measurement errors or unidentified streams. Required for scaling up lab-scale to plant-scale.
- **CoT-CP mapping**: Sister to conservation laws (B3) and stoichiometry (C3). For combinatorial problems, "mass balance" = total count is conserved across decompositions. Already covered.
- **Connects to**: B3, C3.

## G4. Process Flow Diagram (PFD) / Piping & Instrumentation Diagram (P&ID)

- **Source**: ANSI/ISA-S5.1-1984 (R1992), *Instrumentation Symbols and Identification*.
- **Practice**: Visual representations of a chemical process: nodes are unit operations, edges are streams, instrumentation symbols mark control loops. Standardized symbols allow engineers worldwide to read a P&ID without ambiguity.
- **CoT-CP mapping**: Standardize the *output format* of CoT to be a structured graph: nodes = intermediate quantities, edges = derivations, with specific symbols for assumptions, given facts, theorem applications, computational steps. Different from chain-of-custody (NON_AI §L1) — CoC is annotation; PFD is structural diagram. Tools like `tikzcd` could render the resulting graph for paper figures.
- **Connects to**: Chain of custody; structured CoT.
- **Cost/Value**: 1-2 weeks. Value: low-med — useful for paper figures and interpretability research.

---

# Top-12 actionable picks

Rank-ordered by (value × tractability) for v1 paper or near-term v2:

1. **A1. Pólya's Look Back step** (HIGH value, 2-3 days). Mandatory verification primitive. Substitution + special cases + units check. Binary signal → CRC.
2. **F3. Kalman filter for per-step CP** (HIGH value, 2 weeks). Formalizes our heuristic running threshold; principled per-step Bayesian update.
3. **B2. Fermi estimation as sanity check** (HIGH value, 3-5 days). Catches gross arithmetic errors via cheap consistency check.
4. **E2. Adaptive-N via mesh-refinement convergence** (HIGH value, 1 week). Dynamic SC sample count via stability criterion.
5. **A4. Tao heuristic-rigorous mode separation** (med-high, 1 week). Explicit two-mode CoT with cross-check.
6. **B6. Blind analysis protocol** (med-high, 0 cost, document only). Reviewer-credibility move.
7. **G1. HAZOP guidewords as systematic verifier** (med-high, 1-2 weeks). Most thorough probing protocol available.
8. **A2. Lakatos dialectic** (med-high, 1 week). Adversarial proof refinement, structured.
9. **E4. Critical Path Method for trace** (med-high, 2 weeks). Formalizes earliest-bad-step intuition (HGJ Idea 1.1).
10. **G2. LOPA layered-protection §5 analysis** (med, 3-5 days). Clean diagnostic figure.
11. **A6. Lean-verifiable proof subset** (HIGH value if feasible, 2-3 weeks). Ground-truth verification for autoformalizable problems.
12. **B1+B3+B4 Sanity-check primitive library** (med, 1 week, bundled). Dimensional + conservation + boundary cases as a single library.

---

# Items deferred to v2 / future work

- A3 (Erdős Book proofs) — interesting framing, small lift
- A5 (reverse mathematics) — citation-validity score, narrow
- C1 (retrosynthesis backward chaining) — formal-proof analog, expensive
- C2 (multi-spectroscopy) — generalizes existing `tiebreak_lex`
- D1 (distance ladder calibration) — extension of Theorem 3, paper-quality
- D2 (multi-wavelength) — sister to C2
- E1 (Factor of Safety framing) — pure framing
- E3 (tolerance stacking) — pure framing
- E5 (Weibull / reliability) — analysis-side
- F1 (Hamming codes / numeric+fractional redundancy) — narrow
- F2 (lock-in amplifier paraphrase) — refines Pilot M
- F4 (ROC/AUC reporting) — presentation
- F5 (PLL for cadence control) — narrow
- G3 (mass balance) — covered
- G4 (P&ID structured graph) — figures-side

---

# Bibliographic backbone (the 25 most-citable sources)

| # | Source |
|---|---|
| 1 | Pólya, G. (1945/2014), *How to Solve It*, Princeton |
| 2 | Lakatos, I. (1976), *Proofs and Refutations*, Cambridge |
| 3 | Aigner, M. & Ziegler, G. M. (2018), *Proofs from THE BOOK*, 6th ed., Springer |
| 4 | Tao, T., *What's New* blog & career advice pages |
| 5 | Simpson, S. G. (2009), *Subsystems of Second Order Arithmetic*, Cambridge |
| 6 | de Moura, L. et al. (2021), *Lean 4*, CADE 28 |
| 7 | Buckingham, E. (1914), Phys Rev 4: 345 — π theorem |
| 8 | Weinstein, L. & Adam, J. A. (2008), *Guesstimation*, Princeton |
| 9 | Noether, E. (1918), *Invariante Variationsprobleme* — conservation |
| 10 | Boas, M. L. (2005), *Mathematical Methods in the Physical Sciences*, Wiley |
| 11 | CERN Courier (multiple), *Five sigma revisited* |
| 12 | Klein, J. R. & Roodman, A. (2005), Annu Rev Nucl Part Sci 55:141 — blind analysis |
| 13 | Corey, E. J. (1967), Pure & Appl Chem 14:19 — retrosynthesis |
| 14 | Silverstein, R. M. et al. (2014), *Spectrometric Identification*, Wiley |
| 15 | Brown, T. L. et al. (2017), *Chemistry: The Central Science*, Pearson |
| 16 | Riess, A. G. et al. (2022), ApJL 934:L7 — distance ladder |
| 17 | ASCE 7-22 (2022), *Minimum Design Loads*, ASCE |
| 18 | Cook, R. D. et al. (2002), *Concepts and Applications of FEA*, Wiley |
| 19 | Bjørke, Ø. (1989), *Computer-Aided Tolerancing*, ASME |
| 20 | Kelley, J. E. & Walker, M. R. (1959), Eastern Joint Comp Conf — CPM |
| 21 | Weibull, W. (1951), J Appl Mech 18:293 — Weibull distribution |
| 22 | Hamming, R. W. (1950), Bell Sys Tech J 29:147 — error-correcting codes |
| 23 | Kalman, R. E. (1960), J Basic Eng 82:35 — Kalman filter |
| 24 | Marcum, J. I. (1947), RAND RM-754 — ROC curves |
| 25 | AIChE CCPS (2001), *Layer of Protection Analysis* — LOPA |

---

# How this folds into existing project documents

| Existing doc | Update |
|---|---|
| `theorems/PAPER_OUTLINE.md` | §6 "future work" subsection: A6 Lean verification, F3 Kalman filtering, D1 distance-ladder calibration |
| `theorems/THEOREM_REVIEW.md` | Add B6 (blind analysis) as a reproducibility recommendation |
| `score_ideation/synthesis/RESULTS_SUMMARY_KR.md` | New "Wave 5 brainstorm" section: A1 Look Back, B2 Fermi check, E2 mesh-refinement adaptive-N, F3 Kalman-Approach-A |
| `experiments/src/` | Implement A1 (Look Back primitive), B2 (Fermi-rigorous cross-check), B1+B3+B4 (sanity-check library) — all cheap |
| `paper §6 discussion` | Frame CoT-CP as the *statistical-discipline* analog of these *engineering-discipline* sanity checks; concrete table mapping practices to score functions |
| `paper §5 figures` | F4 ROC/AUC presentation; G2 LOPA-style layered-accuracy ablation |

---

# Companion files

- `CROSS_DISCIPLINARY_IDEAS.md` — stat / info-theory / numerical analysis (AI/ML adjacent) — kept
- `NON_AI_DOMAIN_PRACTICES.md` — medicine / aviation / law / military / surgery (operational-discipline) — kept
- `MATH_SCI_ENG_PRACTICES.md` — this file (mathematics / pure science / engineering)
- `papers/ANALYSIS.md` — within-AI literature deep synthesis
- `papers/RELATED_WORK_DRAFT.md` — §2 prose
- `papers/INDEX.md`, `papers/positioning_matrix.md`, `papers/references.bib` — paper catalog

---

## Source URLs

- [How to Solve It — Wikipedia](https://en.wikipedia.org/wiki/How_to_Solve_It)
- [Pólya's Strategy — LibreTexts Math](https://math.libretexts.org/Courses/Coalinga_College/Math_for_Educators_(MATH_010A_and_010B_CID120)/05:_Problem_Solving/5.02:_George_Polya's_Strategy)
- [Tao "Solving mathematical problems" — career page](https://terrytao.wordpress.com/career-advice/solving-mathematical-problems/)
- [Buckingham π theorem — Wikipedia](https://en.wikipedia.org/wiki/Buckingham_pi_theorem)
- [MIT 2.25 Buckingham π notes](https://ocw.mit.edu/courses/2-25-advanced-fluid-mechanics-fall-2013/c0a4521f55e9191d557c167e99e97469_MIT2_25F13_The_Buckingham.pdf)
- [Yale astrophysics Buckingham π notes](http://www.astro.yale.edu/coppi/astro520/buckingham_pi/Buckinghamforlect1.pdf)
- [CERN Courier — Five sigma revisited](https://cerncourier.com/a/five-sigma-revisited/)
- [CERN FAQ on five sigma](https://home.cern/resources/faqs/five-sigma)
- [HAZOP — Wikipedia](https://en.wikipedia.org/wiki/Hazard_and_operability_study)
- [HAZOP overview — SafetyCulture](https://safetyculture.com/topics/hazop)
- [HAZOP — University of Michigan SAFEChE tutorial](https://safeche.engin.umich.edu/tutorials/hazop-tutorial/)
- [Adobe Five Whys explainer](https://business.adobe.com/blog/basics/5-whys-root-cause-analysis)
- [Riess+ 2022, Distance Ladder & Hubble Constant](https://iopscience.iop.org/article/10.3847/2041-8213/ac5c5b)
- [Klein & Roodman 2005, Blind analysis review](https://www.annualreviews.org/doi/10.1146/annurev.nucl.55.090704.151521)
