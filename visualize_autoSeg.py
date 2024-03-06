import cv2
import numpy as np
import os
# polygon_dir = 'D:\\2PAPER_DATA\\ADNOC Subset\\ADNOC 5002_Segmentation\\labels'
polygon_dir = r'D:\2PAPER_DATA\S_images\images\labels'  # load the .txt auto-seg file's root
# image_dir = 'D:\\2PAPER_DATA\\ADNOC Subset\\ADNOC 5002_Segmentation\\images\\images'
image_dir = r'D:\2PAPER_DATA\S_images\images'  # load the image root
output_dir = r'D:\2PAPER_DATA\S_images\images\imageMasks'  # outputs images with mask
output_dir1 = r'D:\2PAPER_DATA\S_images\images\masks'  # outputs masks only
# output_dir = 'D:\\2PAPER_DATA\\ADNOC Subset\\ADNOC 5002_Segmentation\\Image_with_mask'  # 存储处理后图像的目录
# output_dir1 = 'D:\\2PAPER_DATA\\ADNOC Subset\\ADNOC 5002_Segmentation\\mask'

# 函数用于绘制多边形掩码并保存图像，同时添加标签
def draw_and_save_mask_with_labels(image_path, labels_and_polygons, output_dir):
    # 读取图像
    image = cv2.imread(image_path)
    for labels, polygons in labels_and_polygons:
        for label in labels:
            label = int(label)
            if label == 0:
                color = (0, 0, 255)  # 红色
            elif label == 1:
                color = (255, 0, 0)  # 蓝色
        mask = np.zeros_like(image, dtype=np.uint8)

        for polygon_str in polygons:  # polygons is a list with n elements, to enumerate
            polygon_coords = [float(coord) for coord in polygon_str.split()]  # select the float
            num_points = len(polygon_coords) // 2
            polygon_coords = [(int(polygon_coords[i] * (image.shape[1])), int(polygon_coords[i+1] * (image.shape[0]))) for i in range(0, num_points * 2, 2)]

            pts = np.array(polygon_coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            # 绘制多边形填充
            cv2.fillPoly(mask, [pts], color)  # red
        # 在掩码上添加标签文本
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(mask, str(label), (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        masked_image = cv2.addWeighted(image, 1, mask, 0.5, 0)
        # 保存结果图像
        result_file = os.path.splitext(os.path.basename(image_path))[0] + f'_label_{label}_mask.jpg'
        result_path = os.path.join(output_dir, result_file)
        cv2.imwrite(result_path, masked_image)

        # 保存掩码图像
        mask_file = os.path.splitext(os.path.basename(image_path))[0] + '_' + str(label) + '_mask.jpg'
        mask_path = os.path.join(output_dir1, mask_file)
        cv2.imwrite(mask_path, mask)

        # map_image = cv2.addWeighted(image, 1, mask, 0.5, 0)
        # map_file = os.path.splitext(os.path.basename(image_path))[0] + '_' + str(label) + '_mapImg.jpg'
        # map_path = os.path.join(output_dir, map_file)
        # cv2.imwrite(map_path, map_image)


# 获取所有的.txt文件
txt_files = [f for f in os.listdir(polygon_dir) if f.endswith('.txt')]
# image_files = [z for z in os.listdir(image_dir) if z.endswith('.jpg')]
# 遍历.txt文件
for txt_file in txt_files:  # read all polygon.txt files and go through
    # 构建多边形文件的完整路径
    polygon_path = os.path.join(polygon_dir, txt_file)
    print('poly', polygon_path)
    # 读取多边形文件内容
    with open(polygon_path, 'r') as file:
        lines = file.readlines()  # read document line-by-line
    # 解析标签和多边形数据
    labels_and_polygons = []
    current_label = []
    current_polygons = []

    image_file = os.path.splitext(txt_file)[0] + '.jpg'
    # text file in format 'name.txt', fetch the name,for formatting 'name.jpg' # image and txt are then matched
    image_path = os.path.join(image_dir, image_file)
    # image_file = os.path.splitext(polygon_file)[0] + '.jpg'
    print('image', image_path)

    for line in lines:  # lines is a list of n 'label+polygons', and be truncated
        line = line.strip()  # first line only read one 'label+polygons'
        if line:  # now is to read label and polygons
            if ' ' in line:
                # 将行数据解析为浮点数坐标
                polygon_coords = line.split()  # current polygons can be label + polygons
                current_polygons.append(' '.join(polygon_coords[1:]))  # there's no label 0 info.
                current_label.append(' '.join(polygon_coords[0]))
                labels_and_polygons.append((current_label, current_polygons))
                draw_and_save_mask_with_labels(image_path, labels_and_polygons, output_dir)
                # current_label = int(line)  # 标签是整数
                # current_polygons = []
            # else:
            #     if current_label is not None and current_polygons:
            #         labels_and_polygons.append((current_label, current_polygons))
            #     current_label = int(line)  # 标签是整数
            #     current_polygons = []




print('successful')
        # # 添加最后一个标签和多边形
        # if current_label is not None and current_polygons:
        #     labels_and_polygons.append((current_label, current_polygons))
        # # 构建相应的图像文件路径





#####################################
# import cv2
# import numpy as np
# import os
#
#
# # 示例：解析YOLO格式的多边形文件
# def read_yolo_polygons(file_path):
#     polygons = []
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#         for line in lines:
#             # 每行可能包含多个多边形坐标
#             coordinates = line.strip().split()
#             if len(coordinates) % 2 != 0:
#                 continue  # 忽略不完整的坐标行
#             polygon = [float(coord) for coord in coordinates]
#             polygons.append(polygon)
#     return polygons
#
# # 函数用于绘制多边形并保存图
# def draw_and_save_image(image_path, polygons, output_dir):
#     # 读取图像
#     image = cv2.imread(image_path)
#
#     # 绘制多边形
#     for polygon in polygons:
#         points = np.array(polygon, dtype=np.int32).reshape((-1, 2))
#         cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)  # 绘制多边形轮廓
#
#     # 从输入图像文件路径中提取文件名
#     file_name = os.path.basename(image_path)
#
#     # 构建输出图像文件路径
#     output_path = os.path.join(output_dir, file_name)
#
#     # 保存图像
#     cv2.imwrite(output_path, image)
#
# # 指定包含多边形文件和图像文件的目录
# polygons_dir = 'D:/2PAPER_DATA/ADNOC Subset/ADNOC 5002/ADNOC 5002 Segmentation/labels'
# images_dir = 'D:/2PAPER_DATA\ADNOC Subset\ADNOC 5002\ADNOC 5002 Segmentation\images'
# output_dir = 'D:/2PAPER_DATA/ADNOC Subset/ADNOC 5002/ADNOC 5002 Segmentation/visualize'  # 存储处理后图像的目录
#
# # 获取多边形文件列表
# polygon_files = os.listdir(polygons_dir)
#
# # 遍历多边形文件
# for polygon_file in polygon_files:
#     # 构建多边形文件的完整路径
#     polygon_path = os.path.join(polygons_dir, polygon_file)
#
#     # 解析多边形数据，根据需要修改
#     polygons = [(x1, y1, x2, y2, x3, y3, x4, y4) for x1, y1, x2, y2, x3, y3, x4, y4 in read_yolo_polygons(polygon_path)]
#
#     # 构建相应的图像文件路径
#     image_file = os.path.splitext(polygon_file)[0] + '.jpg'
#     image_path = os.path.join(images_dir, image_file)
#
#     # 绘制多边形并保存图像
#     draw_and_save_image(image_path, polygons, output_dir)
#
# print('批量绘制和保存完成。')
#
##############################################在图像上加mask###########################################################
##############################################adding mask to images###################################################
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 读取图像
# image = cv2.imread('your_image.jpg')  # 替换为你的图像文件路径
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像从BGR转换为RGB
#
# # 读取分割掩码（假设是二值掩码，白色表示分割区域）
# mask = cv2.imread('your_mask.png', cv2.IMREAD_GRAYSCALE)  # 替换为你的分割掩码文件路径
# mask = (mask > 0).astype(np.uint8)  # 将掩码转换为二值掩码
#
# # 将分割掩码叠加到原始图像上
# segmented_image = cv2.addWeighted(image, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), 0.3, 0)
#
# # 可视化原始图像和分割结果
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title('Original Image')
#
# plt.subplot(1, 2, 2)
# plt.imshow(segmented_image)
# plt.title('Segmentation Mask Overlay')
# plt.show()
