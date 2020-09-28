rm -r docs/*
pdoc --html  --force --output-dir docs --template-dir template geosardine
mv docs/geosardine/* docs
rmdir docs/geosardine