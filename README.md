# Setup

## Getting Started

Follow these steps to set up your environment and run the project:

### 1. Start Interactive GPU Session
Start an interactive GPU session by typing in the terminal:
```bash
a100sh
```
This will open a new session where you have access to GPUs.

### 2. Load Python Version
Load Python 3.11 using the module system:
```bash
module load python3/3.11
```

### 3. Create Virtual Environment
Create a virtual environment called `.venv`:
```bash
python -m venv .venv
```

Activate the virtual environment:
```bash
source .venv/bin/activate
```

### 4. Install Requirements
Install the required packages into the environment:
```bash
pip install -r requirements.txt
```

### 5. Branch Management
Create and switch to your own development branch:
```bash
git checkout -b your-name
```

If you need to update your branch with the latest changes from main while discarding all your local commits and changes:
```bash
# Switch to your branch (if not already on it)
git checkout your-name

# Reset your branch to match main exactly, discarding all local changes
git fetch origin
git reset --hard origin/main
```

**Warning**: The `git reset --hard origin/main` command will permanently delete all your local commits and changes on this branch. Make sure to backup any important work before running this command.
