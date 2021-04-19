# coding: utf-8
# Created by luuil@outlook.com at 3/30/2021
import os
import numpy as np

from data.dataset_eye import EyeDataset


class MouthDataset(EyeDataset):
    pose_channels = 11

    def __init__(self, pkl_file, root_dir, is_train=True, debug=False):
        super(MouthDataset, self).__init__(pkl_file, root_dir, is_train, debug)
        self.name = "MouthDataset"


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    dataset = MouthDataset(
        pkl_file=r'G:\projects\0results\cartoon_controllable\mouth\mouth_train\ffhq2.2-0_mouth_crop.pkl',
        root_dir=r'G:\projects\0results\cartoon_controllable\mouth\mouth_train\ffhq2.2-0_mouth_crop',
        debug=True
    )
    selected_sample = dataset[100]
    fig = plt.figure()

    ax = plt.subplot(1, 2, 1)
    plt.tight_layout()
    ax.set_title(f'rest: {selected_sample["rest_name"]}')
    plt.imshow(selected_sample["rest"])

    ax = plt.subplot(1, 2, 2)
    plt.tight_layout()
    ax.set_title(f'gt: {selected_sample["expressive_name"]}')
    plt.imshow(np.asarray(selected_sample["expressive"]))

    plt.show()

    print(f'pose={selected_sample["pose"]}')
