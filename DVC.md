Install and Initialize DVC
# pip install dvc
# dvc init

If  dataset is already tracked by SCM (e.g. Git), you can remove it from Git, then add to DVC.
# git rm --cached assignment1/liver_disease.csv
# git commit -m "Stop tracking dataset in git"
# dvc add assignment1/liver_disease.csv
# git add assignment1/.gitignore assignment1/liver_disease.csv.dvc

Create dataset version with DVC
# dvc remote add -d main versions
# dvc push
# git add --all
# git commit -m "Start dataset versioning with DVC - v1.0"
# git push origin main

Tag the version for checkout
# git tag v1.0
# git push --tag
