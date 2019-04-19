import SimpleITK as sitk
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy.io as scio
from skimage import transform
from xlrd import open_workbook
import random

produce = 9  # 还需产生几张
ilegal = [34, 64, 268, 269, 284]
surfacethreshold = 0  # 与空气的阈值
percent_of_node = 0.99
cut_bias = 10
mask_nose = [64, 58, 62]
mask_eye1 = [42, 77, 47]
mask_eye2 = [86, 77, 47]
mask_jaw = [64, 22, 53]
mask = np.array([mask_nose, mask_eye2, mask_eye1, mask_jaw, [0, 0, 0]])
Image_size = [128, 128, 64]


def show3D(list, point):
    fig = plt.figure()
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(list[0, :], list[1, :], list[2, :], color='b', marker='.')
    ax.scatter(point[:, 0], point[:, 1], point[:, 2], color='k', marker='*')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()


def locateSurface(vol):  # dicom x,y,z
    # ———————————————————— show ——————————————————————————————————
    # vol = to_image(vol)
    # for i in range(0, int(vol.shape[1]), 10):
    #     img = vol[:, i, :]
    #     cv2.imshow("1", img)
    #     cv2.waitKey(200)
    a = vol.copy()
    list = []
    map = []   # yz  yz面的哪个位置已经标记过了
    for i in range(0, a.shape[0]):  # x
        if map.__len__() > int(a.shape[1]*a.shape[2]*percent_of_node):
            break
        a_2d = a[i, :, :]
        zy = np.argwhere(a_2d > surfacethreshold)
        for k in zy:
            if not map.__contains__([k[0], k[1]]):
+-----  2 lines: list.append([i, k[0], k[1]])
    # # print(list)  # x y z
    return list


def cutImage(vol, excel_col):
    x_nose = int(excel_col[2]-cut_bias)
    x_nose = max(x_nose, 0)
    x_ear = int(excel_col[3]+cut_bias)
    x_ear = min(x_ear, vol.shape[1])
    y_ear = int(excel_col[4]-cut_bias)
    y_ear = max(y_ear, 0)
    y_ear2 = int(excel_col[5]+cut_bias)
    y_ear2 = min(y_ear2, vol.shape[2])
    z_down = int(excel_col[6]-cut_bias)
    z_down = max(z_down, 0)
    z_up = int(excel_col[7]+cut_bias)
    z_up = min(z_up, vol.shape[0])  # vol dicom z, x, y
    A = vol[z_down:z_up, x_nose:x_ear, y_ear:y_ear2]  # cut
    A = np.swapaxes(A, 0, 1)
    A = np.swapaxes(A, 1, 2)  # dicom x y z
    base_xyz = np.array([x_nose, y_ear, z_down])

    return [A, base_xyz]


def to_image(vol):
    vol = vol.astype(np.float)
    vol = vol - vol.min()
    vol = ((vol - vol.min()) * 255.0 / vol.max()).astype(np.uint8)
    return vol

def face_alignment(vol, col_ROI, base_xyz):
# —————————————————————————————— 读ROI 读入为 dicom坐标——————————————————————————————
    nose_y = col_ROI[2]
    nose_x = col_ROI[3]
    nose_z = col_ROI[4]
    jaw_y = col_ROI[5]
    jaw_x = col_ROI[6]
    jaw_z = col_ROI[7]
    eye1_y = col_ROI[8]
    eye1_x = col_ROI[9]
    eye1_z = col_ROI[10]
    eye2_y = col_ROI[11]  # unuse
    eye2_x = col_ROI[12]
    eye2_z = col_ROI[13]
    eye2 = np.array([eye2_x, eye2_y, eye2_z])
    eye2 = eye2 - base_xyz
    nose = np.array([nose_x, nose_y, nose_z])   # 减base
    nose = nose - base_xyz
    eye1 = np.array([eye1_x, eye1_y, eye1_z])
    eye1 = eye1 - base_xyz
    jaw = np.array([jaw_x, jaw_y, jaw_z])
    jaw = jaw - base_xyz
    src = np.array([nose, eye1, eye2, jaw, [0, 0, 0]])  # dicom xyz ->mask -z x y
    src[:, [0, 1]] = src[:, [1, 0]]  # ->mask x -z y
    src[:, [1, 2]] = src[:, [2, 1]]  # ->mask x y -z
    # src[0:2, 2] = vol.shape[0] - src[0:2, 2]  # ->mask x y z
# —————————————————————————————— 导入所有需要变换的坐标 ——————————————————————————————
    idx = np.array(locateSurface(vol))  # dicom x y z  ->mask -z x y
    idx[:, [0, 1]] = idx[:, [1, 0]]     # ->mask x -z y
    idx[:, [1, 2]] = idx[:, [2, 1]]  #  dicom y z x ->mask x y -z
    a = np.transpose(idx)
    b = np.ones(idx.shape[0])
    c = np.row_stack((a, b))
    # show3D(c, src)  # 显示变换前
# —————————————————————————————— 3D求仿射变换矩阵 + 应用变换 ——————————————————————————————
    T_matrix = np.zeros((3, 4))
    inliers = np.zeros((1, 2))
    cv2.estimateAffine3D(src, mask, T_matrix, inliers)

# —————————————————————————————— 变换所有点 ——————————————————————————————
    dst = np.dot(T_matrix, c)
    # show3D(dst, mask)  # 显示变换后
    dst = np.transpose(dst)  # 每一行xyz
    CT_value = np.zeros(dst.shape[0])
    for ii in range(0, dst.__len__()):
        CT_value[ii] = vol[idx[ii, 2], idx[ii, 0], idx[ii, 1]]
    dst = np.column_stack((dst, CT_value))
    return dst
    # print(dst)

def loadDicom():
    path = '/home/jwzhu/pycharm/CT'
    xl_path = '/home/jwzhu/pycharm/CT_mark.xls'
    ThreeDPoint_path = '/home/jwzhu/pycharm/CT_3D_xyz'
    P_list = os.listdir(ThreeDPoint_path)
    r_xls = open_workbook(xl_path)  # 读取excel文件
    table_cut = r_xls.sheets()[0]
    table_ROI = r_xls.sheets()[1]
    row = table_cut.nrows  # excel的第几行
    label = 491
    pat = 0
    dicom_path = path + '/491/0'
    if os.path.isdir(dicom_path):  # 是dicom文件夹
        # read dicom
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        vol = sitk.GetArrayFromImage(image)  # dicom z, x, y
# —————————————————————————————————— read excel and cut image ——————————————————————————————————
        for r in range(1, row):
            col_cut = table_cut.row_values(r)
            col_ROI = table_ROI.row_values(r)
            if int(col_cut[0])==label and int(col_cut[1])==int(pat):
                break
        [vol, base_xyz] = cutImage(vol, col_cut) # cut image  vol: dicom xyz
        k = 6
        if str(label) + '_' + str(k)+'.txt' in P_list:  # 是都已经生成了
            exit(0)
        col_ROI2 = col_ROI[:]
        bais = random.randint(-5, 6)
        if bais == 0:
            bais = 2
        col_ROI2[k+1] = col_ROI2[k+1] + bais
        dst = face_alignment(vol, col_ROI2, base_xyz)  # face_alignment
# —————————————————————————————————— write file ——————————————————————————————————
#                 txt_path = ThreeDPoint_path + '/' + str(label) + '_' + str(pat) + '.txt'
        txt_path = ThreeDPoint_path + '/' + str(label) + '_' + str(k) + '.txt'
        np.savetxt(txt_path, dst)
        print(label, k, col_cut[0], col_cut[1])


loadDicom()
