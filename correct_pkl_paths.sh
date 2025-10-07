#!/bin/bash

# Directory containing your run folders
BASE_DIR="pkls/AIStats/rigid_eval"

cd "$BASE_DIR" || exit

for dir in */; do
    # Remove trailing slash
    dirname="${dir%/}"

    # Remove the _level=NUMBER part
    newname=$(echo "$dirname" | sed -E 's/_level=[0-9]+//g')

    if [[ "$dirname" != "$newname" ]]; then
        echo "Renaming: $dirname -> $newname"
        mv "$dirname" "$newname"
    fi
done
