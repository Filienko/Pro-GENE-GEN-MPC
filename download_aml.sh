wget https://dl.cispa.de/s/paWjy4mzE5zkzM5/download/aml_data.zip -O data/aml/aml_data.zip
unzip data/aml/aml_data.zip -d data/aml/
mv data/aml/aml_data/* data/aml/
rm -rf data/aml/aml_data
rm -rf data/aml/aml_data.zip 
rm -rf data/aml/__MACOSX/