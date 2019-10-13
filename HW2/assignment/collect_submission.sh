rm -f cs7643_hw2.zip
jupyter-nbconvert --to pdf "*.ipynb"
zip -r cs7643_hw2.zip . -x "*.ipynb_checkpoints*" "*README.md" "*collect_submission.sh" "*.pdf" ".env/*" "*.pyc" "__pycache__/*"
