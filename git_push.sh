#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash git_push.sh <remote_url> [branch] [commit_message] [--force]
#
# Example (SSH):
#   bash git_push.sh git@github.com:<user>/<repo>.git main "initial commit"
#   bash git_push.sh git@github.com:<user>/<repo>.git main "replace remote with local" --force
#
# Example (HTTPS):
#   bash git_push.sh https://github.com/<user>/<repo>.git main "update code"

REMOTE_URL="${1:-}"
BRANCH="${2:-main}"
COMMIT_MSG="${3:-chore: update project files}"
PUSH_MODE="${4:-}"

if [[ -z "${REMOTE_URL}" ]]; then
  echo "Usage: bash git_push.sh <remote_url> [branch] [commit_message] [--force]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is not installed."
  exit 1
fi

if [[ ! -d ".git" ]]; then
  echo "[INFO] Initializing git repository in: ${SCRIPT_DIR}"
  git init
fi

if ! git config user.name >/dev/null 2>&1; then
  echo "[WARN] git user.name is not set. Set it with:"
  echo "       git config user.name \"Your Name\""
fi
if ! git config user.email >/dev/null 2>&1; then
  echo "[WARN] git user.email is not set. Set it with:"
  echo "       git config user.email \"you@example.com\""
fi

if git remote get-url origin >/dev/null 2>&1; then
  CURRENT_REMOTE="$(git remote get-url origin)"
  if [[ "${CURRENT_REMOTE}" != "${REMOTE_URL}" ]]; then
    echo "[INFO] Updating origin URL:"
    echo "       ${CURRENT_REMOTE} -> ${REMOTE_URL}"
    git remote set-url origin "${REMOTE_URL}"
  fi
else
  echo "[INFO] Adding remote origin: ${REMOTE_URL}"
  git remote add origin "${REMOTE_URL}"
fi

echo "[INFO] Staging files..."
git add -A

if git diff --cached --quiet; then
  echo "[INFO] No staged changes. Nothing to commit."
else
  echo "[INFO] Creating commit: ${COMMIT_MSG}"
  git commit -m "${COMMIT_MSG}"
fi

echo "[INFO] Switching branch: ${BRANCH}"
git checkout -B "${BRANCH}"

echo "[INFO] Pushing to origin/${BRANCH}..."
if [[ "${PUSH_MODE}" == "--force" ]]; then
  echo "[WARN] Force push enabled: remote branch will be overwritten to match local snapshot."
  git push -u origin "${BRANCH}" --force
else
  git push -u origin "${BRANCH}"
fi

echo "[DONE] Push completed."
