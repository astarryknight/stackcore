#!/bin/sh

#process arguments
OPTSTRING="v:rc:"
RELEASE=false
COMMIT=false
CMT_MESSAGE=""

while getopts ${OPTSTRING} opt; do
  case ${opt} in
    v)
      VERSION=${OPTARG}
      ;;
    r)
      RELEASE=true
      ;;
    c)
      COMMIT=true
      CMT_MESSAGE=${OPTARG}
      ;;
    ?)
      echo "Invalid option: -${OPTARG}."
      exit 1
      ;;
  esac
done

# #version bumping
VERSION_OLD=$(sed -n '7p' pyproject.toml | awk -F'"' '{print $2}')

if [ -z "${VERSION+xxx}" ]; then
    maj=$(echo $VERSION_OLD | cut -d '.' -f 1)
    min=$(echo $VERSION_OLD | cut -d '.' -f 2)
    bug=$(echo $VERSION_OLD | cut -d '.' -f 3)
    VERSION="$maj.$min.$(($bug+1))"
fi

echo $VERSION

#replace version number in toml file
sed -i.bu "7s/.*/version = \"$VERSION\"/" pyproject.toml

#move old builds from dist to archive
mv dist/* archive

#building
python3 -m pip install --upgrade build
python3 -m build

#uploading to pypi
python3 -m pip install --upgrade twine
python3 -m twine upload --repository testpypi dist/* --verbose


#create github release
if [ "$RELEASE" = true ]; then
    gh release create v$VERSION ./dist/*.gz --latest -n "# stackcore $VERSION Release Notes:" -t "v$VERSION ($(date '+%m/%d/%Y'))"
fi

if [ "$COMMIT" = true ]; then
  git add -A
  echo $OPTARG
  git commit -m "v${VERSION}: ${CMT_MESSAGE}"
  git push
fi