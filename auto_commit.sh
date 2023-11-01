#!/bin/bash
for i in {1..4};do pre-commit run --all-files;done
# git submodule update --recursive
git pull
git add .
read MESSAGE
git commit -m "$MESSAGE"
git push
echo "Done"
read -p "Press any key to continue..."
