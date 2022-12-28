
# import required module
from pathlib import Path
 
# get the path/directory
folder_dir = '/home/torquehq/torquehq-io/Github/Torque-AI/Users_slab/test/a1'
 
# iterate over files in
# that directory
images = Path(folder_dir).glob('*.jpg')
for image in images:
    print(image)