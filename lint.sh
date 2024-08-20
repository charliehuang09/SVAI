find . -type f -name "*.py" | xargs pylint 
find . -name '*.py' -print0 | xargs -0 yapf -i