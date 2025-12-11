#!/bin/bash
# Script to set up a new GitHub repository with the current repo's history

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <new-repo-name>"
    echo "Example: $0 my-new-optim-repo"
    exit 1
fi

NEW_REPO_NAME="$1"
CURRENT_USER=$(git config user.name)
CURRENT_EMAIL=$(git config user.email)

echo "=== Setting up new repository: $NEW_REPO_NAME ==="
echo "Current author: $CURRENT_USER <$CURRENT_EMAIL>"
echo ""

# Check if we're in a git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Show current remote
echo "Current remote:"
git remote -v
echo ""

# Instructions
echo "=== Instructions ==="
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: $NEW_REPO_NAME"
echo "   - Choose public/private"
echo "   - DO NOT initialize with README, .gitignore, or license"
echo "   - Click 'Create repository'"
echo ""
echo "2. After creating the repo, run these commands:"
echo ""
echo "   # Add new remote (replace YOUR_USERNAME with your GitHub username)"
echo "   git remote set-url origin https://github.com/YOUR_USERNAME/$NEW_REPO_NAME.git"
echo ""
echo "   # Push all branches and tags"
echo "   git push -u origin --all"
echo "   git push -u origin --tags"
echo ""
echo "3. To verify only one contributor shows up:"
echo "   - Check: https://github.com/YOUR_USERNAME/$NEW_REPO_NAME/graphs/contributors"
echo "   - All commits should show as: $CURRENT_USER"
echo ""

