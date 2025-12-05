#!/bin/bash
# Script to rewrite Git history with a single author and prepare for new repo

set -e

# Configuration
NEW_AUTHOR_NAME="JG"
NEW_AUTHOR_EMAIL="janghanem@gmail.com"
OLD_AUTHOR_EMAIL="55719558+ghanemja@users.noreply.github.com"

echo "=== Rewriting Git History ==="
echo "Changing all commits to: $NEW_AUTHOR_NAME <$NEW_AUTHOR_EMAIL>"
echo ""

# Create a backup branch first
echo "Creating backup branch..."
git branch backup-before-rewrite

# Rewrite commit history
echo "Rewriting commit history..."
git filter-branch --env-filter "
    export GIT_COMMITTER_NAME='$NEW_AUTHOR_NAME'
    export GIT_COMMITTER_EMAIL='$NEW_AUTHOR_EMAIL'
    export GIT_AUTHOR_NAME='$NEW_AUTHOR_NAME'
    export GIT_AUTHOR_EMAIL='$NEW_AUTHOR_EMAIL'
" --tag-name-filter cat -- --branches --tags

echo ""
echo "=== History Rewritten ==="
echo "All commits now have author: $NEW_AUTHOR_NAME <$NEW_AUTHOR_EMAIL>"
echo ""
echo "Next steps:"
echo "1. Create a new repository on GitHub with your desired name"
echo "2. Run the following commands:"
echo "   git remote set-url origin <NEW_REPO_URL>"
echo "   git push -u origin --all"
echo "   git push -u origin --tags"
echo ""
echo "To restore the original history:"
echo "   git reset --hard backup-before-rewrite"

