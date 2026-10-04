# COMP-672 Python Labs

Student laboratories for John Glossner’s COMP-672 course. Start with a lab’s **Open in Colab** button. Viewing these public materials requires no GitHub account or invitation; sign in to a Google account to run Colab and save your own notebook in Drive.

Each lab is worth 2 points and has a handout, an unfilled notebook, and an equivalent Python starter. Choose one starter format. All 24 exercise functions remain intentionally unimplemented.

## Lab index

| Lab | Topic | Files | Notebook |
| --- | --- | --- | --- |
| 1 | Tokenization and a smoothed bigram model | [Handout](labs/LAB01.md) · [Python](labs/week01_starter.py) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week01_starter.ipynb) |
| 2 | Logistic regression embeddings and a neural classifier | [Handout](labs/LAB02.md) · [Python](labs/week02_starter.py) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week02_starter.ipynb) |
| 3 | A tiny transformer with causal and masked objectives | [Handout](labs/LAB03.md) · [Python](labs/week03_starter.py) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week03_starter.ipynb) |
| 4 | Retrieval and a bounded evidence-based controller | [Handout](labs/LAB04.md) · [Python](labs/week04_starter.py) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week04_starter.ipynb) |
| 5 | A small recurrent translator and beam search | [Handout](labs/LAB05.md) · [Python](labs/week05_starter.py) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week05_starter.ipynb) |
| 6 | Speech features CTC and recognition error analysis | [Handout](labs/LAB06.md) · [Python](labs/week06_starter.py) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week06_starter.ipynb) |
| 7 | Residual audio quantization and a TTS evaluation harness | [Handout](labs/LAB07.md) · [Python](labs/week07_starter.py) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/glossner/COMP-672-Labs/blob/main/labs/week07_starter.ipynb) |

## Start in Google Colab

1. Open the lab using its badge and sign in to Google.
2. Select **File → Save a copy in Drive** before editing. Rename your copy, for example `COMP672_Lab01_YourName.ipynb`. Work in this copy.
3. Choose a CPU runtime. Run the setup and environment cells, read the embedded handout, and complete the functions marked `NotImplementedError`. Add the required comparisons and analysis.
4. Run your completed notebook from top to bottom and keep its outputs. The supplied checks are examples, not an exhaustive grader. An untouched starter intentionally stops at `NotImplementedError`.

Every notebook contains its handout, setup, data generation, starter code, blank analysis areas, and results export. No clone, Drive mount, API key, external dataset, or pretrained model is needed. Opening the notebook does not run it, save answers to GitHub, or share your work. You control execution, saving, and sharing.

## Save and submit

1. Complete the notebook, all required experiments, and the three brief analysis responses. Fill in configuration, versions, seed, split policy, settings, runtimes, and result labels.
2. Run the JSON export cell. Download `weekNN_results.json` through Colab’s Files sidebar. It also prints the JSON into notebook output. Runtime files are temporary and are **not** included just because you share the notebook.
3. Use **File → Save** on your Drive copy and wait for saving to finish. Keep **Edit → Notebook settings → Omit code cell output when saving this notebook** turned off. Reopen the copy and confirm your code, analysis, and outputs are saved.
4. On your copy choose **Share → General access → Anyone with the link → Viewer**. Click **Copy link**, then **Done**. Verify the link opens for a signed-out viewer. Anyone with the link can read the saved code, output, and comments, so include only your course submission.
5. Submit **your completed copy’s link**, the downloaded results JSON, and the execution method through the course channel specified by John Glossner. For Colab, the execution method is “Run cells from top to bottom in a fresh CPU runtime.” Do not submit the blank starter’s GitHub or Colab link.

If your school Google account blocks “Anyone with the link,” contact the instructor for an approved submission alternative. No automatic sharing, emails, permission changes, or submission occurs in these notebooks.

**Cloning alone does not save answers.** A clone is a local copy of repository files; changes must be saved locally and, for a GitHub workflow, explicitly committed and pushed. A Colab runtime is temporary. Saving a notebook in Drive preserves its cells and outputs, not the runtime’s files or installed libraries.

## Update a notebook already copied to Drive

The original release limited Python to 3.11/3.12 and pinned NumPy 2.0.2, which lacks Python 3.13 wheels. This revision accepts Python 3.13 and uses compatible package pins. A Drive copy is independent: refreshing it does not pull updates from GitHub.

- **If you have not started:** reopen the lab using the README’s Colab badge, reload that GitHub-backed tab if already open, and choose **File → Save a copy in Drive** again.
- **If you have work to preserve:** keep the old Drive copy as a backup. Save a fresh copy from the updated badge and transfer your completed code, comparisons, and analysis into it. Alternatively, replace only the two code cells under **Install the lab environment** (setup and environment) and the setup instructions with those from the latest version of the **same lab**. Labs 3 and 5 include PyTorch setup. Leave exercise and analysis cells intact.
- Save your copy, select **Runtime → Restart session**, and rerun the setup and environment cells. The setup should now accept Python 3.13. After completing and rerunning the lab, save outputs and submit the link to the copy you actually worked in.

## Optional GitHub workflow

A GitHub account is needed only if you choose to fork and save there. Fork the repository into your account, then save your work explicitly to your fork (Colab: **File → Save a copy to GitHub**, which asks for GitHub authorization), or commit and push your local edits. Opening a notebook from GitHub does not automatically update GitHub. Forks of this public repository are public; follow the instructor’s policy about public coursework. The Drive save/share workflow above is the default and avoids collecting students’ GitHub IDs. Do not submit answers as a pull request to the course repository.

## Environment and local use

Use Colab’s **default Python CPU runtime**; no Python downgrade or past runtime selection is needed. The labs support Python **3.11–3.13**, with all three versions covered by CI. The shared packages have exact direct-version pins in `requirements.txt`. Labs 3 and 5 additionally use CPU PyTorch 2.11.0. The notebooks install the same versions themselves when you run setup. No GPU or paid Colab plan is required by the lab code; Colab availability and runtime limits still apply.

On 2026-10-04, Google’s [backend snapshot](https://github.com/googlecolab/backend-info/blob/e39694e267a4b13c71d339c9fa7d4552f7995ac7/os-info.txt) lists Python 3.13; its [package snapshot](https://github.com/googlecolab/backend-info/blob/e39694e267a4b13c71d339c9fa7d4552f7995ac7/pip-freeze.txt) lists the versions used here. These snapshots can lag production, so the setup cell prints your actual Python version and the environment cell checks the imported package versions. Run setup before imports; if prompted after a package change, choose **Runtime → Restart session**, then rerun from setup. Binary wheels are required; setup will not compile packages from source. Do not replace Colab’s system Python. Record the printed versions in your submission.

For a local Linux/Windows environment, from this repository:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-torch.txt
python check_environment.py
python labs/week01_starter.py
```

On Windows use `py -3.13 -m venv .venv` and `.venv\Scripts\activate` (Command Prompt). On macOS install `torch==2.11.0` from PyPI instead of `requirements-torch.txt`. To edit notebooks locally, additionally install `jupyterlab` and run `jupyter lab`. These platform alternatives are not part of the Linux smoke test.

An unfinished starter raises `NotImplementedError`. Finish the required functions before running its full checks and experiment. For a source submission, save the result dictionary with `json.dump`, convert any arrays to lists, and include configuration and the actual execution command. If linear algebra is unexpectedly slow, set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before starting Python.

## Data and scope

All included text, labels, signals, and simulated ratings are original generated instructional examples already supplied with the student labs. No external course assets are required. Lab 2 uses a known negation rule. Lab 3 samples a finite grammar whose sequences can recur across splits. Lab 5 holds out combinations in an invented reordering task. Lab 6 uses synthetic tones and supplied recognition hypotheses, not a trained ASR system. Lab 7 uses scalar residual quantization and simulated ratings, not a speech synthesizer or human listening study. Interpret results within these limits.

## Repository validation

The release check validates notebook schema and syntax, empty outputs, the 24 exercise gaps, and notebook/script consistency. Smoke checks on Python 3.11, 3.12, and 3.13 load the provided scaffolding, generate synthetic data, exercise the scientific/ML APIs used by the scaffolding, and confirm that unfinished checks stop as expected. They do not fill answers or run completed experiments.

```bash
python -m pip install -r requirements-dev.txt
python scripts/validate.py
python scripts/validate.py --smoke
```

The GitHub Actions workflow runs these checks on repository changes; it does not run or share students’ notebooks. Setup and notebook-kernel smoke tests also passed locally on Python 3.13.13. These are local/CI compatibility tests, not execution in a Google-hosted Colab runtime. Students still run, save, and share their own notebooks.

## Official help

Behavior checked against primary documentation on 2026-10-04:

- [Google’s Colab/GitHub guide](https://github.com/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb): opening notebooks, saving copies, and badge URL format.
- [Colab FAQ](https://research.google.com/colaboratory/faq.html): Drive storage, shared notebook contents, temporary runtimes, and setup cells.
- [Google Drive sharing](https://support.google.com/drive/answer/2494822?hl=en): Anyone with the link and Viewer permissions.
- [Colab backend information](https://github.com/googlecolab/backend-info): current Python and package snapshots.
- [NumPy 2.1 release notes](https://numpy.org/doc/2.1/release/2.1.0-notes.html): Python 3.13 support.
- [Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html): explains that the past-version list is separate from the current default runtime.
- [GitHub repositories](https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories): public access and cloning.
- [GitHub forks](https://docs.github.com/en/pull-requests/reference/forks): public fork visibility.
