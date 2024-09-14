# OVER 50 mb do not include in git push

find . -type f -size +50M > large_files.txt

sed -i 's/^\.\///' large_files.txt
sed -i 's|^|/skydata2/dylanelliott/letkf-hybrid-speedy/|' large_files.txt

# remove from git index 
while IFS= read -r file; do
    git rm --cached "$file"
done < large_files.txt

echo "DONE"
