#!/usr/bin/env bash
# Wipes what the station setup installed on this computer, so the setup can be run again from scratch.
#
#   cp ~/physical-ai-hacks/scripts/wipe_station.sh /tmp/ && cd ~ && bash /tmp/wipe_station.sh
#   bash /tmp/wipe_station.sh --apt    # also removes the apt packages the setup installs
#
# Run it from a copy outside the repo, as above: it deletes the repo. It lists what it found and removes nothing until
# you type "wipe". Kept on purpose: the SSH server (you would lose access to this computer), git, and anything the
# setup does not add. The Dream platform keeps the station's registration and the API key: remove those on the
# platform's pages if you want them gone too.
set -uo pipefail

with_apt=0
[ "${1:-}" = "--apt" ] && with_apt=1

if [ "$(id -u)" = 0 ] || [ -z "${HOME:-}" ] || [ "$HOME" = "/" ]; then
  echo "run this as the station's own user (not root, not with sudo); it asks for sudo itself where needed"
  exit 1
fi
running=$(pgrep -af 'dhk-collect|dreamcli|check_cameras|check_arms|check_quest' | grep -v "wipe_station" || true)
if [ -n "$running" ]; then
  echo "stop these first (Ctrl+C in their terminals, or 'dreamcli robot down'):"
  echo "$running" | sed 's/^/  /'
  exit 1
fi

home_paths=(
  "$HOME/physical-ai-hacks"
  "$HOME/dream-env"
  "$HOME/.dreamlabs"
  "$HOME/hackathon_data"
  "$HOME/.cache/dream_hackathon"
  "$HOME/.cache/dream_ik"
  "$HOME/.cache/uv"
  "$HOME/.local/share/uv"
  "$HOME/.local/bin/uv"
  "$HOME/.local/bin/uvx"
  "$HOME/.local/bin/env"
  "$HOME/.local/bin/env.fish"
  "$HOME/.android"
  "$HOME/.git-credentials"
)
system_files=(/etc/udev/rules.d/90-dreamlabs-can.rules /etc/udev/rules.d/90-can.rules)
apt_packages=(build-essential python3-dev curl adb tmux git-lfs python3-pip)
profiles=("$HOME/.bashrc" "$HOME/.profile" "$HOME/.bash_profile" "$HOME/.zshrc" "$HOME/.zshenv")
uv_line='. "$HOME/.local/bin/env"'

echo "This removes, from this computer:"
found=()
for path in "${home_paths[@]}"; do
  if [ -e "$path" ]; then
    found+=("$path")
    echo "  $(du -sh "$path" 2>/dev/null | cut -f1)	$path"
  fi
done
system_found=()
for file in "${system_files[@]}"; do
  if [ -e "$file" ]; then
    system_found+=("$file")
    echo "  (sudo)	$file"
  fi
done
profile_found=()
for profile in "${profiles[@]}"; do
  if [ -f "$profile" ] && grep -qxF "$uv_line" "$profile"; then
    profile_found+=("$profile")
    echo "  (line)	uv's PATH line in $profile"
  fi
done
has_lfs=0
if command -v git >/dev/null && git config --global --get filter.lfs.process >/dev/null 2>&1; then
  has_lfs=1
  echo "  (git)	git-lfs's settings in ~/.gitconfig"
fi
has_quest_app=0
if command -v adb >/dev/null && adb devices 2>/dev/null | grep -q 'device$'; then
  has_quest_app=1
  echo "  (adb)	the teleop app on the connected Quest"
fi
in_video=0
if id -nG "$USER" | tr " " "\n" | grep -qx video; then
  in_video=1
  echo "  (group)	$USER's membership of the video group"
fi
apt_found=()
if [ "$with_apt" = 1 ]; then
  for package in "${apt_packages[@]}"; do
    if dpkg -s "$package" >/dev/null 2>&1; then
      apt_found+=("$package")
    fi
  done
  [ "${#apt_found[@]}" -gt 0 ] && echo "  (apt)	${apt_found[*]}"
fi
if [ "${#found[@]}" -eq 0 ] && [ "${#system_found[@]}" -eq 0 ] && [ "${#profile_found[@]}" -eq 0 ] \
  && [ "$has_lfs" = 0 ] && [ "$has_quest_app" = 0 ] && [ "$in_video" = 0 ] && [ "${#apt_found[@]}" -eq 0 ]; then
  echo "  nothing: this computer has no station setup left"
  exit 0
fi
echo "Kept: the SSH server, git, and everything else."
read -r -p "Type wipe to remove all of the above: " answer
if [ "$answer" != "wipe" ]; then
  echo "nothing removed"
  exit 0
fi

if [ "$has_quest_app" = 1 ]; then
  adb uninstall com.rail.oculus.teleop >/dev/null 2>&1 && echo "removed the teleop app from the Quest"
fi
for path in "${found[@]}"; do
  rm -rf -- "$path" && echo "removed $path"
done
for profile in "${profile_found[@]}"; do
  sed -i "\\|^\\. \"\\\$HOME/\\.local/bin/env\"\$|d" "$profile" && echo "removed uv's PATH line from $profile"
done
if [ "$has_lfs" = 1 ]; then
  git lfs uninstall >/dev/null 2>&1 || git config --global --remove-section filter.lfs 2>/dev/null
  echo "removed git-lfs's settings from ~/.gitconfig"
fi
if [ "$in_video" = 1 ]; then
  sudo gpasswd -d "$USER" video >/dev/null && echo "removed $USER from the video group"
fi
if [ "${#system_found[@]}" -gt 0 ]; then
  sudo rm -f -- "${system_found[@]}" && echo "removed ${system_found[*]}"
  sudo udevadm control --reload-rules
fi
if [ "${#apt_found[@]}" -gt 0 ]; then
  sudo apt-get remove -y "${apt_found[@]}" && sudo apt-get autoremove -y && echo "removed ${apt_found[*]}"
fi
echo "done. Reboot so the CAN links get their default names back, then start the setup again from the runbook."
echo "On the Dream platform you may also revoke the API key and remove this station's registration."
