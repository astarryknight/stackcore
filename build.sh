#!/bin/sh

#version bumping
VERSION_OLD=$(sed -n '7p' pyproject.toml | awk -F'"' '{print $2}')

if [[ -n "$1" ]]; then
    VERSION=$1
else
    maj=$(echo $VERSION_OLD | cut -d '.' -f 1)
    min=$(echo $VERSION_OLD | cut -d '.' -f 2)
    bug=$(echo $VERSION_OLD | cut -d '.' -f 3)
    VERSION="$maj.$min.$(($bug+1))"
fi

sed -i.bu "7s/.*/version = \"$VERSION\"/" pyproject.toml

#move old builds from dist to archive
mv dist/* archive

#building
python3 -m pip install --upgrade build
python3 -m build

#uploading to pypi
python3 -m pip install --upgrade twine
python3 -m twine upload --repository testpypi dist/* --verbose