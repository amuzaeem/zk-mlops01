Install and Initialize DVC
# pip install dvc
# dvc init

If  dataset is already tracked by SCM (e.g. Git), you can remove it from Git, then add to DVC.
# git rm --cached assignment1/liver_disease.csv
# git commit -m "Stop tracking dataset in git"
# dvc add assignment1/liver_disease.csv

Commit the changes
# git add assignment1/.gitignore assignment1/liver_disease.csv.dvc
# git commit -m "Track dataset with DVC"
# git push origin main
