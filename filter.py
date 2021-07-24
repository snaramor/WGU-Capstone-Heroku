#
# WGU C964 Capstone Project
# Equipment Faults in Manufacturing Environments
# Sean Naramor
# July 23, 2021
#
####################################
# This function was used to filter out the files relevant to the process we are measuring
####################################
def filter_data():
    import os
    import shutil

    path = os.path.dirname(__file__) + '/'

    source_path = path + 'data_raw/'
    target_path = path + 'data_filtered/'

    file_iterator = os.scandir(source_path)
    count = 0
    for file in file_iterator:
        with open(os.path.join(source_path, file), 'r', encoding='utf8') as f:
            file_content = f.read()
            # The process, also called a recipe, that we are looking for is referred to by the number 7988
            if file_content.find('\\AG\\RECIPES\\7988.V00') >= 0:
                count += 1
                src = source_path + file.name
                dest = target_path + file.name
                shutil.copyfile(src, dest)

    print(f'Found {count} files...')

