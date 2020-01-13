#!/bin/bash
# https://www.howtogeek.com/howto/30184/10-ways-to-generate-a-random-password-from-the-command-line/
TOKEN=$(tr -cd '[:alnum:]' < /dev/urandom | fold -w32 | head -n1)
printf "#\n"
printf "# Data Science 4 Effective Operations\n"
printf "#\n"
printf "# starting jupyter notebook&lab ... \n"
echo "$TOKEN" > token.txt
jupyter notebook --ip 0.0.0.0 --port 9999 --NotebookApp.token="$TOKEN" &> notebook.log &
jupyter lab --ip 0.0.0.0 --port 9998 --NotebookApp.token="$TOKEN" &> lab.log &
sleep 3
printf "done\n"

PUBLIC_IP=$(curl -s icanhazip.com)
LAB_LINK="http://%s:9999/?token=c10c04d634cc2bdf3b9a5a45485e682e1ffbc21a75f88f97"
printf "#\n"
printf "# Notebook:\n"
printf '# * local url: http://%s:9999/?token=%s\n' "0.0.0.0" "$TOKEN"
printf '# * public url: http://%s:9999/?token=%s\n' "$PUBLIC_IP" "$TOKEN"
printf "#\n"
printf "# Lab:\n"
printf '# * local url: http://%s:9998/?token=%s\n' "0.0.0.0" "$TOKEN"
printf '# * public url: http://%s:9998/?token=%s\n' "$PUBLIC_IP" "$TOKEN"
printf "#\n"

bash
