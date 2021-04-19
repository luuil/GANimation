import os
# import argparse
# import glob
import cv2
# from utils import face_utils
from utils import cv_utils, util
# import face_recognition
from PIL import Image
import torchvision.transforms as transforms
import torch
# import pickle
import numpy as np
from models.models import ModelsFactory
from options.test_options import TestOptions
from moviepy.editor import ImageSequenceClip
from abc import abstractmethod


class MorphInTheWild:
    eye_pose_names = [
        "iris_rotation_x",
        "iris_rotation_y",
        "iris_small",
        'eye_wink',
        'eye_happy_wink',
        'eye_relaxed',
        'eye_unimpressed',
        'eye_raised_lower_eyelid',
        'eye_surprised',
    ]
    eye_pose_value_ranges = [(-1, 1)] * 2 + [(0, 1)] * 7
    mouth_pose_names = [
        "mouth_aaa",
        "mouth_iii",
        "mouth_uuu",
        "mouth_eee",
        "mouth_ooo",
        "mouth_delta",
        "mouth_smirk",
        "mouth_lowered_corner_left",
        "mouth_lowered_corner_right",
        "mouth_raised_corner_left",
        "mouth_raised_corner_right",
    ]
    mouth_pose_values = [(0, 1)] * len(mouth_pose_names)

    def __init__(self, opt):
        self._opt = opt
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._model.set_eval()
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        self._mean = mean
        self._std = std
        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize(64, transforms.InterpolationMode.BICUBIC),
                                              # transforms.RandomHorizontalFlip(p=1),
                                              transforms.Normalize(mean=mean, std=std),
                                              ])
        self._de_transform = transforms.Compose([
            transforms.Normalize(mean=[-m / s for m, s in zip(self._mean, self._std)], std=[1. / s for s in self._std]),
            transforms.ToPILImage(),
            transforms.Resize(85, transforms.InterpolationMode.BICUBIC),
        ])

    def morph_file(self, img_path, gt_path, expresion):
        img = cv_utils.read_cv2_img(img_path)
        gt = cv_utils.read_cv2_img(gt_path)
        # morphed_img = self._img_morph(img, expresion)
        morphed_img = self.morph_face(img, gt, expresion)
        output_name = f'{os.path.basename(img_path)}_{np.random.randint(0, 100)}_out.png'
        self.save_img(morphed_img, output_name)

    # def _img_morph(self, img, expresion):
    #     bbs = face_recognition.face_locations(img)
    #     if len(bbs) > 0:
    #         y, right, bottom, x = bbs[0]
    #         bb = x, y, (right - x), (bottom - y)
    #         face = face_utils.crop_face_with_bb(img, bb)
    #         face = face_utils.resize_face(face)
    #     else:
    #         face = face_utils.resize_face(img)
    #
    #     morphed_face = self._morph_face(face, expresion)
    #
    #     return morphed_face

    def morph_face(self, rest, gt, expresion):
        rest = torch.unsqueeze(self._transform(Image.fromarray(rest)), 0)
        gt = torch.unsqueeze(self._transform(Image.fromarray(gt)), 0)
        expresion = torch.unsqueeze(torch.from_numpy(expresion), 0)
        test_batch = {
            "rest": rest,
            "expressive": gt,
            "pose": expresion,
            "rest_name": "rest",
            "expressive_name": "expressive",
        }
        self._model.set_input(test_batch)
        imgs, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        return imgs['concat']

    def save_img(self, img, filename):
        util.mkdir(self._opt.output_dir)
        filepath = os.path.join(self._opt.output_dir, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img)

    @staticmethod
    def show_img(img, winname="img"):
        cv2.imshow(winname, img)
        cv2.waitKey(0)


class MorphTestData(MorphInTheWild):
    def __init__(self, opt):
        super(MorphTestData, self).__init__(opt)

        from data.custom_dataset_data_loader import CustomDatasetDataLoader
        data_loader_test = CustomDatasetDataLoader(self._opt, is_for_train=False)
        self.data_loader_test = data_loader_test.load_data()

    def morph(self, n: int):
        for i_batch, batch in enumerate(self.data_loader_test):
            if i_batch > n:
                break
            pose = batch['pose'].cpu().float().numpy()[0]
            if np.all(pose == 0):  # skip all zero
                continue

            self._model.set_input(batch)
            imgs, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)

            if self._opt.dataset_mode == 'eye':
                pose_names = self.eye_pose_names
            elif self._opt.dataset_mode == 'mouth':
                pose_names = self.mouth_pose_names
            else:
                raise Exception(f'not supported: {self._opt.dataset_mode}')
            idx = np.argmax(pose != 0)
            name = f'{pose_names[idx]}_{pose[idx]:.03f}.png'
            img_concat = imgs['concat']
            self.save_img(img_concat, name)


class MorphFaceInTheWild(MorphInTheWild):
    def __init__(self, opt):
        super(MorphFaceInTheWild, self).__init__(opt)

    @staticmethod
    def blend(img, region, roi):
        x, y, w, h = roi
        region = cv2.resize(region, (w, h), interpolation=cv2.INTER_CUBIC)
        img[y:y + h, x:x + w] = region

    def save_imgs_as_video(self, imgs, filename, fps):
        util.mkdir(self._opt.output_dir)
        clip = ImageSequenceClip(imgs, fps=fps)
        # clip.write_gif(os.path.join(self._opt.output_dir, filename))
        clip.write_videofile(os.path.join(self._opt.output_dir, filename))
        clip.close()

    def morph(self, rest, pose):
        rest = torch.unsqueeze(self._transform(Image.fromarray(rest)), 0)
        pose = torch.unsqueeze(torch.from_numpy(pose), 0)
        imgs = self._model.forward_one(rest, pose)
        return imgs

    def rnd_poses(self, n: int = 10, fps=30, pose_names=MorphInTheWild.eye_pose_names,
                  pose_values=MorphInTheWild.eye_pose_value_ranges):
        poses = dict()
        for idx, name in enumerate(pose_names):
            value_range = pose_values[idx]
            prefix_values = [value_range[0]] * fps
            suffix_values = [value_range[1]] * fps
            middle_values = np.linspace(value_range[0], value_range[1], n)
            values = np.concatenate([prefix_values, middle_values, suffix_values], axis=0)
            poses_cur = list()
            for v in values:
                pose = np.zeros(len(pose_names), dtype=np.float32)
                pose[idx] = v
                poses_cur.append(pose)
            poses[name] = poses_cur

        return poses

    @staticmethod
    @abstractmethod
    def crop(img, idx=0):
        pass

    @abstractmethod
    def morph_face_from_file(self, img_path, pose):
        pass

    @abstractmethod
    def morph_face_from_file_video(self, img_path, n_each_pose=60, fps=30):
        pass


class MorphEyesInTheWild(MorphFaceInTheWild):
    def __init__(self, opt):
        super(MorphEyesInTheWild, self).__init__(opt)

    @staticmethod
    def crop(img, idx=0):
        rois_left = [
            (256, 190, 85, 85),
            (540, 320, 85, 85),
            (540, 300, 85, 85),
        ]
        rois_right = [
            (256 - 85, 190, 85, 85),
            (410, 320, 85, 85),
            (400, 300, 85, 85),
        ]
        roi_l, roi_r = rois_left[idx], rois_right[idx]

        x, y, w, h = roi_l
        crop_left = img[y:y + h, x:x + w]
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0))

        x, y, w, h = roi_r
        crop_right = img[y:y + h, x:x + w]
        return crop_left, roi_l, crop_right, roi_r

    def morph_face_from_file(self, img_path, pose):
        img = cv_utils.read_cv2_img(img_path)

        crop_left, roi_l, crop_right, roi_r = self.crop(img)

        imgs = self.morph(crop_left, pose)
        morphed = imgs['result']
        self.blend(img, morphed, roi_l)
        # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0))

        output_name = f'{os.path.basename(img_path)}_{np.random.randint(0, 100)}_out.png'
        self.save_img(img, output_name)

    def morph_face_from_file_video(self, img_path, n_each_pose=60, fps=30):
        img = cv_utils.read_cv2_img(img_path)
        _w_s, h_s, _c = img.shape

        crop_left, roi_l, crop_right, roi_r = self.crop(img, idx=0)

        morphed_imgs = list()
        t0 = util.time_now()
        cnt = 0
        poses_dict = self.rnd_poses(n_each_pose, fps, self.eye_pose_names, self.eye_pose_value_ranges)
        for name in poses_dict:
            cnt += len(poses_dict[name])
            for pose in poses_dict[name]:
                morphed_left = self.morph(crop_left, pose)['result']
                if name == 'iris_rotation_y':
                    pose *= -1
                morphed_right = self.morph(crop_right, pose)['result']
                img_cp = img.copy()
                self.blend(img_cp, morphed_left, roi_l)
                self.blend(img_cp, morphed_right, roi_r)

                cv2.putText(img_cp, f"{name}={pose[np.argmax(pose != 0)]:.03f}", (10, h_s - 20),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0))

                morphed_imgs.append(img_cp)

        t_avg = util.time_diff_ms(t0) / (cnt * 2)
        print(f"infer: {t_avg:.03f}ms")
        output_name = f'{os.path.basename(img_path)[:-4]}.mp4'
        # output_name = f'{os.path.basename(img_path)}' \
        #     f'_{name}_n{len(poses_dict[name])}_{"left" if left else "right"}.mp4'
        self.save_imgs_as_video(morphed_imgs, output_name, fps)


class MorphMouthInTheWild(MorphFaceInTheWild):
    def __init__(self, opt):
        super(MorphMouthInTheWild, self).__init__(opt)

    @staticmethod
    def crop(img, idx=0):
        rois_left = [
            # (230, 275, 55, 55),  # val/image2(1637)_output.png
            (230, 270, 60, 60),  # val/image2(2163)_output.png image3(661)_output
        ]
        roi_l = rois_left[idx]

        x, y, w, h = roi_l
        crop_left = img[y:y + h, x:x + w]
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0))

        return crop_left, roi_l

    def morph_face_from_file(self, img_path, pose):
        img = cv_utils.read_cv2_img(img_path)

        crop, roi = self.crop(img)

        imgs = self.morph(crop, pose)
        morphed = imgs['result']
        self.blend(img, morphed, roi)

        output_name = f'{os.path.basename(img_path)}_{np.random.randint(0, 100)}_out.png'
        self.save_img(img, output_name)

    def morph_face_from_file_video(self, img_path, n_each_pose=60, fps=30):
        img = cv_utils.read_cv2_img(img_path)
        _w_s, h_s, _c = img.shape

        crop, roi = self.crop(img, idx=0)

        morphed_imgs = list()
        t0 = util.time_now()
        cnt = 0
        poses_dict = self.rnd_poses(n_each_pose, fps, self.mouth_pose_names, self.mouth_pose_values)
        for name in poses_dict:
            cnt += len(poses_dict[name])
            for pose in poses_dict[name]:
                morphed = self.morph(crop, pose)['result']
                img_cp = img.copy()
                self.blend(img_cp, morphed, roi)

                cv2.putText(img_cp, f"{name}={pose[np.argmax(pose != 0)]:.03f}", (10, h_s - 20),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0))

                morphed_imgs.append(img_cp)

        t_avg = util.time_diff_ms(t0) / cnt
        print(f"infer: {t_avg:.03f}ms")
        output_name = f'{os.path.basename(img_path)[:-4]}.mp4'
        # output_name = f'{os.path.basename(img_path)}' \
        #     f'_{name}_n{len(poses_dict[name])}_{"left" if left else "right"}.mp4'
        self.save_imgs_as_video(morphed_imgs, output_name, fps)


def main():
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    morph = MorphInTheWild(opt)

    image_path = opt.input_path
    gt_path = opt.gt_path
    # expression = np.random.uniform(0, 1, opt.cond_nc)
    expression = np.zeros(opt.cond_nc)
    expression[0] = 0.2
    morph.morph_file(image_path, gt_path, expression)


def main_test_dataset():
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    morph = MorphTestData(opt)
    morph.morph(100)


def main_eye_wild():
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    morph = MorphEyesInTheWild(opt)

    image_path = opt.input_path
    # expression = np.random.uniform(0, 1, opt.cond_nc)
    expression = np.zeros(opt.cond_nc)
    expression[morph.eye_pose_names.index('eye_wink')] = 0.5
    # expression[morph.pose_names.index('iris_rotation_x')] = 0.5
    # expression[morph.pose_names.index('iris_rotation_y')] = 0.5
    expression[morph.eye_pose_names.index('iris_small')] = 0.5
    morph.morph_face_from_file(image_path, expression)


def main_eye_wild_rnd_poses():
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    morph = MorphEyesInTheWild(opt)

    image_path = opt.input_path
    morph.morph_face_from_file_video(image_path, n_each_pose=60, fps=30)


def main_mouth_wild_rnd_poses():
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    morph = MorphMouthInTheWild(opt)

    image_path = opt.input_path
    morph.morph_face_from_file_video(image_path, n_each_pose=10, fps=30)


if __name__ == '__main__':
    # main()
    # main_test_dataset()
    # main_eye_wild()
    # main_eye_wild_rnd_poses()
    main_mouth_wild_rnd_poses()
