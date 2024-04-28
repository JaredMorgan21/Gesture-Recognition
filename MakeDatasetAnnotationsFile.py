import os


class_names_to_idxs = {name:idx for idx, name in enumerate(os.listdir('train'))}
idxs_to_class_names = {idx:name for idx, name in enumerate(os.listdir('train'))}

top_level_dir = 'train'

with open('labels_file', 'w') as f:
    for batch_folder in os.listdir(top_level_dir):
        print(batch_folder)
        for image_name in os.listdir(os.path.join(top_level_dir, batch_folder)):
            f.write(os.path.join(top_level_dir, batch_folder, image_name) + ', ' + str(class_names_to_idxs[batch_folder]) +'\n')
            print(os.path.join(top_level_dir, batch_folder, image_name) + ', ' + str(class_names_to_idxs[batch_folder]) +'\n')

class_names_to_idxs = {name: idx for idx, name in enumerate(os.listdir('test'))}
idxs_to_class_names = {idx: name for idx, name in enumerate(os.listdir('test'))}

top_level_dir = 'test'

with open('labels_file', 'a') as f:
    for batch_folder in os.listdir(top_level_dir):
        print(batch_folder)
        for image_name in os.listdir(os.path.join(top_level_dir, batch_folder)):
            f.write(os.path.join(top_level_dir, batch_folder, image_name) + ', ' + str(
                class_names_to_idxs[batch_folder]) + '\n')
            print(os.path.join(top_level_dir, batch_folder, image_name) + ', ' + str(
                class_names_to_idxs[batch_folder]) + '\n')