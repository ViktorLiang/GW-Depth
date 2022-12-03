import os
import json

def id_to_image(imgs, save_file):
    img_names = os.listdir(imgs)
    map_dict = {}
    for i, v in enumerate(img_names):
        map_dict[i] = v
    assert len(img_names) == len(map_dict)
    with open(save_file, 'w') as f:
        json.dump(map_dict, f)
    print('json image_id save to {}'.format(save_file))

    im_names = [im.split('.')[0] for im in img_names]
    save_im = os.path.join(os.path.dirname(save_file), 'eval_nogt_{}.txt'.format(len(im_names)))
    with open(save_im, 'w+') as f:
        for i, im in enumerate(im_names):
            f.write(im+'\n')
    print()
    print('image name save to {}'.format(save_im))


if __name__ == '__main__':
    im_dir = '/home/ly/data/datasets/trans-depth/20221118/20221118_134053_images'
    save_file = '/home/ly/data/datasets/trans-depth/20221118/id_to_image.json'
    id_to_image(im_dir, save_file)