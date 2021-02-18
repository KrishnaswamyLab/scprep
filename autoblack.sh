cat <<EOF >> .git/hooks/pre-commit
#!/bin/sh

set -e

files=\$(git diff --staged --name-only --diff-filter=d -- "*.py")

for file in \$files; do
  black -t py35 -q \$file
  git add \$file
done
EOF
chmod +x .git/hooks/pre-commit
