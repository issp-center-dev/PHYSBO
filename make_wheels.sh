# This file is for the package maintainer

rm -rf build dist *.egg-info

ret=0
python setup.py bdist_wheel || ret=$?

echo 
echo 

if [ $ret -ne 0 ];then
  echo "Failed to build package"
  exit 1
fi

echo 'To upload to PyPI:'
echo 'twine upload --repository pypi dist/*'
