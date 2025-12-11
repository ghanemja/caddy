#!/bin/bash
# Script to push rewritten history to a new GitHub repository

if [ -z "$1" ]; then
    echo "Usage: $0 <github-username>/<new-repo-name>"
    echo "Example: $0 ghanemja/optim-v2"
    exit 1
fi

NEW_REPO="$1"
NEW_REPO_URL="https://github.com/${NEW_REPO}.git"

echo "=== Pushing to New Repository ==="
echo "Repository: $NEW_REPO_URL"
echo ""

# Remove old remote
echo "Removing old remote..."
git remote remove origin 2>/dev/null || true

# Add new remote
echo "Adding new remote: $NEW_REPO_URL"
git remote add origin "$NEW_REPO_URL"

# Push all branches
echo ""
echo "Pushing all branches..."
git push -u origin --all

# Push all tags
echo ""
echo "Pushing all tags..."
git push -u origin --tags

echo ""
echo "=== ✓ Complete ==="
echo "Your repository is now at: https://github.com/$NEW_REPO"
echo "Check contributors: https://github.com/$NEW_REPO/graphs/contributors"

