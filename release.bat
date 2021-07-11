rm -r dist rlgym_compat.egg-info
python setup.py sdist && twine upload dist/*
rm -r dist rlgym_compat.egg-info