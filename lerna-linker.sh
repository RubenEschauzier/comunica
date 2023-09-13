#!/bin/bash
files=''
output='/dev/stdout'
unlink=false

print_usage() {
  printf "Unknown flag. \nFlags: \n -v (Verbose) \n -f File of linked packages \n -l Where to log command output, either .txt file or /dev/null to supress \n"
}

while getopts 'f:o:u' flag; do
  case "${flag}" in
    f) files="${OPTARG}";;
    o) output="${OPTARG}" ;; 
    u) unlink=true ;; 
    *) print_usage
       exit 1 ;;
  esac
done

# Allow all packages to be unlinked
if [ $unlink = true ];
then
    while read p; do
        yarn unlink "$p"
    done <$files
    echo Done!
    exit [0];
fi

if [ $files ];
then
    while read p; do
        yarn link "$p"
    done <$files
    echo Done!

else
    echo "Linking packages in the repository"
    
    # First unlink all packages so we always use the version in this dev comunica setup
    # This should also unlink packages from different folders
    yarn run lerna exec --no-bail -- yarn unlink > "$output"
    yarn run lerna exec --no-bail -- yarn link >> "$output"

    yarn run lerna ls > lerna-ls-output.txt
    grep '^@' lerna-ls-output.txt > packages.txt
    rm lerna-ls-output.txt

    echo Done! 
    echo Run the command: $PWD/lerna-linker.sh -f $(realpath packages.txt) in the repository you want to use this dev version of comunica in.
fi
