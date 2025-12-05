# How to Create a New Repo with Fixed Contributor History

## Problem
GitHub shows 2 contributors because commits have two different author emails:
- `JG <janghanem@gmail.com>` (most commits)
- `ghanemja <55719558+ghanemja@users.noreply.github.com>` (a few commits)

## Solution: Rewrite History and Create New Repo

### Step 1: Rewrite Git History (Unify Authors)

Run this to rewrite all commits to use a single author:

```bash
# Make sure you're on the master branch and have no uncommitted changes
git checkout master
git status  # Should be clean

# Rewrite all commits to use your preferred email
git filter-branch --env-filter '
    export GIT_COMMITTER_NAME="JG"
    export GIT_COMMITTER_EMAIL="janghanem@gmail.com"
    export GIT_AUTHOR_NAME="JG"
    export GIT_AUTHOR_EMAIL="janghanem@gmail.com"
' --tag-name-filter cat -- --branches --tags
```

**Note:** This rewrites history. If you've already pushed, you'll need to force push later.

### Step 2: Create New Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `your-new-repo-name` (e.g., `optim-v2`, `cad-optimizer`, etc.)
3. Choose public/private
4. **DO NOT** initialize with README, .gitignore, or license
5. Click "Create repository"

### Step 3: Push to New Repository

```bash
# Remove old remote
git remote remove origin

# Add new remote (replace YOUR_USERNAME and NEW_REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/NEW_REPO_NAME.git

# Push all branches and tags
git push -u origin --all
git push -u origin --tags
```

### Step 4: Verify

Check the contributors page:
- https://github.com/YOUR_USERNAME/NEW_REPO_NAME/graphs/contributors

You should now see only **1 contributor** (you).

## Alternative: Simpler Method (If you don't need to preserve all history)

If you just want a fresh start with current code:

```bash
# Create a new repo on GitHub (same steps as above)

# Remove old git history
rm -rf .git

# Initialize new repo
git init
git add .
git commit -m "Initial commit"
git branch -M main

# Add new remote and push
git remote add origin https://github.com/YOUR_USERNAME/NEW_REPO_NAME.git
git push -u origin main
```

This loses commit history but ensures only 1 contributor.

## Restore Original History (if needed)

If you created a backup branch before rewriting:
```bash
git reset --hard backup-before-rewrite
```

