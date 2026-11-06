#!/usr/bin/env bash
# Removes the Dream API key from this station once its setup is done.
#
#   bash scripts/remove_api_key.sh
#
# The Dream library reads the key from ~/.dreamlabs/credentials, or from DREAM_API_KEY in the environment. The key step
# also keeps the file it replaced as credentials.bak. This deletes the DREAM_API_KEY line from both files and keeps the
# platform addresses, so a team can later save its own key with the same key step. A script cannot clear a key held by
# a shell or a program that is already running, so it checks for those and says what to do. Revoking the key itself is
# done on the platform's API Keys page.
set -euo pipefail

dir="$HOME/.dreamlabs"
removed=0
for file in "$dir/credentials" "$dir/credentials.bak"; do
  [ -f "$file" ] || continue
  if grep -q '^DREAM_API_KEY=' "$file"; then
    sed -i '/^DREAM_API_KEY=/d' "$file"
    echo "removed the API key from $file"
    removed=1
  fi
done
[ "$removed" = 1 ] || echo "no API key found in $dir"

# Anything left that would still hand the key to the next shell or program (file names only, never the value).
left=$(grep -ls 'DREAM_API_KEY' "$dir"/credentials* "$HOME/.bashrc" "$HOME/.profile" "$HOME/.bash_profile" \
  "$HOME/.zshrc" 2>/dev/null || true)
if [ -n "$left" ]; then
  echo "warning: these files still mention DREAM_API_KEY; remove that line by hand:"
  echo "$left" | sed 's/^/  /'
fi
if [ -n "${DREAM_API_KEY:-}" ]; then
  echo "warning: DREAM_API_KEY is set in this shell; run: unset DREAM_API_KEY"
fi
if pgrep -f dreamcli >/dev/null 2>&1; then
  echo "warning: a dreamcli program is running and keeps the key it loaded until it stops (dreamcli robot down)"
fi
echo "done. If the key is no longer needed, revoke it at https://platform.dreamlabs.ai/api-keys"
