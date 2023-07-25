# realestate-document_classification


cd .venv/lib/python3.9/site-packages
zip -r ../../../../package.zip .

zip -g package.zip main.py

tfenv install latest:^0.15
tfenv use 0.15.5