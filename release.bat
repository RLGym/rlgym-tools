rm -r dist rlgym_tools.egg-info
python setup.py sdist && twine upload dist/*
rm -r dist rlgym_tools.egg-info