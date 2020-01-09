conda create --name maskrcnn python=3.6 -y
pip install -r requirements.txt
pip install pycocotools
cd samsung
python samsung.py train