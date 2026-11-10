// lltRL - Write PPO From Scratch, Then Train With GRPO (LLM Training Lab, module 19, premium, class A).
//
// Lab page (type 'lab-lesson'), rendered by src/components/module/LabLesson.jsx.
// Blueprint: .claude/docs/LLM_TRAINING_PROGRAM_PLAN.md section 10 entry 19, 7.8.4 (research extension: DPO vs PPO frontier),
// section 12. Paired with llm/llmRLHF (recap card). Class A: a narrated overview is built later from
// labs/lltRL/OVERVIEW_SPEC.md, and a Try it widget from labs/lltRL/TRYIT_SPEC.md (data labs/lltRL/tryit_data.json).
// This draft sets neither overview nor lesson.tryIt.
//
// NUMBERS ARE A CONTRACT. Every figure traces to aiml-companion/courses/llm-training/runs/lltRL/ or an exact derivation.
// Code blocks are the files labs/lltRL/steps/cellNN_stepK.py (export_page_code.py builds them in), so the page, the notebook
// and the recorded runs share one source. GPU runs: Google Cloud A100 (llt-c3) with the T4 recipe (fp16 autocast, GradScaler,
// fp16 frozen bases), python 3.12.16, torch 2.11.0+cu130, transformers 5.22.1, trl 1.17.1, peft 0.21.2.
// T4 times on the page are projections until re-timed on a Kaggle T4.
// Code strings: no backslash, no backtick, no dollar-brace and no non-ASCII character.

// @@CODE@@

export const lltRL = @@PAGE@@;
