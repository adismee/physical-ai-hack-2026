#!/usr/bin/env bash
# Lets this station's account bring the YAM arms' CAN links up without a password, so the dashboard's
# "Bring ... up" buttons work. Run it once per station, as the station's own user (it asks for sudo):
#
#   bash scripts/allow_can_bringup.sh                                  # the links can_follower_l and can_follower_r
#   bash scripts/allow_can_bringup.sh can_follower_l can_follower_r    # or name them
#
# It allows exactly `ip link set <link> up type can bitrate 1000000` for those links, and nothing else: no other
# command, no other link, and never taking a link down. scripts/wipe_station.sh removes it again.
set -euo pipefail

rule=/etc/sudoers.d/90-dream-hackathon-can
links=("$@")
[ "${#links[@]}" -gt 0 ] || links=(can_follower_l can_follower_r)
user=$(id -un)
if [ "$(id -u)" = 0 ]; then
  echo "run this as the station's own user (not root, not with sudo); it asks for sudo itself"
  exit 1
fi
ip=$(command -v ip) || { echo "ip is not installed"; exit 1; }

commands=()
for link in "${links[@]}"; do
  if [[ ! $link =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "not a link name: $link"
    exit 1
  fi
  commands+=("$ip link set $link up type can bitrate 1000000")
done

draft=$(mktemp)
trap 'rm -f "$draft"' EXIT
{
  echo "# Written by physical-ai-hacks/scripts/allow_can_bringup.sh: lets $user bring the arms' CAN links up"
  echo "# without a password, for the dashboard's buttons. Nothing else. scripts/wipe_station.sh removes it."
  (IFS=,; echo "$user ALL=(root) NOPASSWD: ${commands[*]}")
} > "$draft"

sudo visudo -cqf "$draft" >/dev/null || { echo "the rule did not pass visudo's check; nothing installed"; exit 1; }
sudo install -m 0440 -o root -g root "$draft" "$rule"
mkdir -p "$HOME/.local/share/dream_hackathon"
echo "$rule" > "$HOME/.local/share/dream_hackathon/can-sudo-rule"  # how scripts/wipe_station.sh knows to remove it
echo "installed $rule:"
sed 's/^/  /' "$draft"  # the installed copy is readable by root only
for link in "${links[@]}"; do
  if sudo -n -l "$ip" link set "$link" up type can bitrate 1000000 >/dev/null 2>&1; then
    echo "check: $user may now bring $link up without a password"
  else
    echo "check FAILED for $link: sudo still asks for a password"
    exit 1
  fi
done
