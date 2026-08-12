# Prompt Log 01 — Part B setup and requirements audit

## Task / prompt

I asked the AI assistant to begin FINS5545 Part B using the official `projectB_starter` folder. I instructed it to read `PROJECT_BRIEF.md` first, work inside that folder, follow the hosted-data workflow, use `scripts/check_handin.py`, keep the GitHub repository private during development, and later deploy the Streamlit app following `docs/STUDENT_DEPLOY.md`. I also asked for step-by-step guidance aimed at a high mark, with financial logic, PyCharm actions, commands, expected outputs, and checks.

## What the AI did

The assistant:
- opened and read `PROJECT_BRIEF.md` before changing the project files;
- inspected the starter folder structure and the provided stubs for portfolios, sentiment, fusion, Streamlit, tests, and hand-in checks;
- read `docs/STUDENT_DEPLOY.md` and the submission checklist;
- renamed/extracted the working folder as `z5646061_projectB` in its working environment;
- ran the baseline hand-in checker and smoke test;
- replaced the placeholder `AGENTS.md` with project-specific instructions based on the brief and my stated working preferences.

## Verification

The baseline `scripts/check_handin.py` run showed that the folder structure and deployment files were present. The only blocking failure was the unchanged placeholder agent file. The remaining items were warnings for outputs and the report that have not yet been built.

The smoke test confirmed that imports resolved. The hosted dataset could not be downloaded in the assistant's sandbox because that environment did not have DNS/network access. This is an environment limitation rather than evidence of a defect in `src/data_access.py`. The hosted data load still needs to be verified locally in PyCharm.

## Correction / decision

I will not change the official data-loading design or manually commit raw data just because the assistant sandbox cannot access the internet. The final project must continue to load raw data through `src/data_access.py`, as required by the brief.

The next development step is to migrate and validate my own Part A ETL/feature foundation into Part B before building the portfolio models.

## Stage 1 handoff

See Prompt Log 02 for the Stage 1 Part A reuse, deterministic headline-ordering
correction, and validation evidence.
