#/bin/bash

echo "Moving templates over"
cd scripts/
cp -r ./*.xctemplate ~/Library/Developer/Xcode/Templates/

echo "Installing git hooks"
cp pre-commit ../.git/hooks/pre-commit
chmod +x ../.git/hooks/pre-commit