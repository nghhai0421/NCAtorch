# Hai's Preferred Coding and Comment Style

These preferences apply across this project.

- Prefer straightforward, sequential code with a workflow readable from top to bottom. Keep ordinary orchestration directly visible.
- Extract functions for substantial logic, useful reuse, or clearer reading. Avoid tiny forwarding helpers, layers of wrappers, and repeatedly passing the same long parameter lists.
- Establish settings and local variables once and reuse them. Group related parameters where useful, keeping dependencies understandable.
- Separate meaningful responsibilities. Add classes, files, and interfaces only when they serve a practical need. Balance modularity with simplicity; avoid both enormous functions and elaborate frameworks.
- Keep experiment entry points, including scripts in `scripts/` and any new `run.py`, readable like experiment scripts with visible stages. Put substantial data preparation in `nca/data/`, architecture in `nca/core/models/`, and training loops in `nca/training/`. For extension-specific work, use the corresponding modules under `nca/extensions/` where appropriate. Follow the existing structure rather than introducing `process.py`, `model.py`, or `train.py` solely to match another repository.
- Use short, standalone stage comments where they help readers follow the workflow, such as `# Process Data`, `# Initiate Model`, `# Load and Train Model`, `# Predict Test`, and `# Save Result`. Adapt the wording to the actual NCA workflow, such as `# Load Config`, `# Create Dataset`, `# Create Model`, `# Train Model`, or `# Run Inference`. Use blank lines for readability.
- Keep comments plain and concise. Explain non-obvious decisions and transformations. Avoid decorative banners, excessive numbering, narration of every statement, and inaccurate descriptions of units.
