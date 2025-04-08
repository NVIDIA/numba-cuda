#!/bin/bash

# pick up files that was changed in the PR
git diff-tree --no-commit-id --name-only HEAD~15..HEAD -r > format_files.txt
# cherry pick formatter configs
git cherry-pick 71124bdaa71d032934016b79de8e688b2e153804
# run formatter on changed files
pre-commit run --files $(cat format_files.txt)
git add $(cat format_files.txt)
rm format_files.txt
git commit --no-verify -m "Apply pre-commit"
# merge origin/main
git merge origin/main
# apply ours changes on conflict files (changes should be pseudo conflicts)
git checkout --ours .
git add .
git merge --continue
git push