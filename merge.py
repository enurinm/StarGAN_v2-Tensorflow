from PIL import Image
import cv2
import os
import sys

def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt' or ext =='.json':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    img_files.sort()
    mask_files.sort()
    gt_files.sort()
    return img_files, mask_files, gt_files

def main():
    O_PATH = './dataset/celebA-HQ_gender/test/'
    N_PATH = './results/StarGAN_v2_celebA-HQ_gender_gan_1adv_1sty_1ds_1cyc/'
    RES = './noised_img/'

    ori_img_lists, _, _ = get_files(O_PATH)
    #noised_img_lists, _, _ = get_files(N_PATH)

    for k, in_path in enumerate(ori_img_lists):
        sys.stdout.write('TEST IMAGES: {:d}/{:d}: {:s} \r'.format(k + 1, len(ori_img_lists), in_path))
        sys.stdout.flush()

        ori_img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        ori_img = cv2.resize(ori_img, dsize=(256, 256))
        img_name = os.path.basename(in_path)
        #h, w, c = img.shape
        noised_img = cv2.imread(N_PATH+img_name, cv2.IMREAD_COLOR)

        print(in_path)
        print(N_PATH+img_name)

        b1, g1, r1=cv2.split(ori_img)
        b2, g2, r2=cv2.split(noised_img)

        original_bias=0.9
        noise_bias=1-original_bias

        newimg = cv2.addWeighted(ori_img, original_bias, noised_img, noise_bias, 0)
        cv2.imwrite(RES + 'img_' + str(k) + '.jpg', newimg)


 
if __name__ == '__main__':
    main()
