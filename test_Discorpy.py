# 使用[Discorpy库]实现光伏图像校正
# 3.3.1
import numpy as np
import discorpy.losa.loadersaver as io
import discorpy.prep.preprocessing as prep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post

#加载图像
# Initial parameters
file_path = "./images/DJI_20210803111219_0313_W.JPG"
output_base = "./correct_image/"
num_coef = 5  # Number of polynomial coefficients
mat0 = io.load_image(file_path) # Load image
(height, width) = mat0.shape

# 提取参考点
# Segment dots
mat1 = prep.binarization(mat0)
# Calculate the median dot size and distance between them.
(dot_size, dot_dist) = prep.calc_size_distance(mat1)
# Remove non-dot objects
mat1 = prep.select_dots_based_size(mat1, dot_size)
# Remove non-elliptical objects
mat1 = prep.select_dots_based_ratio(mat1)
io.save_image(output_base + "/segmented_dots.jpg", mat1) # Save image for checking
# Calculate the slopes of horizontal lines and vertical lines.
hor_slope = prep.calc_hor_slope(mat1)
ver_slope = prep.calc_ver_slope(mat1)
print("Horizontal slope: {0}. Vertical slope {1}".format(hor_slope, ver_slope))

#点分为线
# Group points to horizontal lines
list_hor_lines = prep.group_dots_hor_lines(mat1, hor_slope, dot_dist)
# Group points to vertical lines
list_ver_lines = prep.group_dots_ver_lines(mat1, ver_slope, dot_dist)
# Optional: remove horizontal outliners
list_hor_lines = prep.remove_residual_dots_hor(list_hor_lines, hor_slope)
# Optional: remove vertical outliners
list_ver_lines = prep.remove_residual_dots_ver(list_ver_lines, ver_slope)
# Save output for checking
io.save_plot_image(output_base + "/horizontal_lines.png", list_hor_lines, height, width)
io.save_plot_image(output_base + "/vertical_lines.png", list_ver_lines, height, width)

# Optional: correct perspective effect. Only available from Discorpy 1.4
# list_hor_lines, list_ver_lines = proc.regenerate_grid_points_parabola(
#                      list_hor_lines, list_ver_lines, perspective=True)

#计算分组点与其拟合直线的距离来检查点线的直线度
list_hor_data = post.calc_residual_hor(list_hor_lines, 0.0, 0.0)
list_ver_data = post.calc_residual_ver(list_ver_lines, 0.0, 0.0)
io.save_residual_plot(output_base + "/hor_residual_before_correction.png",
                      list_hor_data, height, width)
io.save_residual_plot(output_base + "/ver_residual_before_correction.png",
                      list_ver_data, height, width)

#计算中心失真(COD) 和后向映射的系数
# Calculate the center of distortion
(xcenter, ycenter) = proc.find_cod_coarse(list_hor_lines, list_ver_lines)
# Calculate coefficients of the correction model
list_fact = proc.calc_coef_backward(list_hor_lines, list_ver_lines,
                                    xcenter, ycenter, num_coef)
# Save the results for later use.
io.save_metadata_txt(output_base + "/coefficients_radial_distortion.txt",
                     xcenter, ycenter, list_fact)
print("X-center: {0}. Y-center: {1}".format(xcenter, ycenter))
print("Coefficients: {0}".format(list_fact))
"""
>> X-center: 1252.1528590042283. Y-center: 1008.9088499595639
>> Coefficients: [1.00027631e+00, -1.25730878e-06, -1.43170401e-08,
                  -1.65727563e-12, 7.89109870e-16]
"""
#使用修正模型的确定参数，我们可以展开点线并检查修正结果
# Apply correction to the lines of points
list_uhor_lines = post.unwarp_line_backward(list_hor_lines, xcenter, ycenter,
                                            list_fact)
list_uver_lines = post.unwarp_line_backward(list_ver_lines, xcenter, ycenter,
                                            list_fact)
# Save the results for checking
io.save_plot_image(output_base + "/unwarpped_horizontal_lines.png", list_uhor_lines,
                   height, width)
io.save_plot_image(output_base + "/unwarpped_vertical_lines.png", list_uver_lines,
                   height, width)
# Calculate the residual of the unwarpped points.
list_hor_data = post.calc_residual_hor(list_uhor_lines, xcenter, ycenter)
list_ver_data = post.calc_residual_ver(list_uver_lines, xcenter, ycenter)
# Save the results for checking
io.save_residual_plot(output_base + "/hor_residual_after_correction.png",
                      list_hor_data, height, width)
io.save_residual_plot(output_base + "/ver_residual_after_correction.png",
                      list_ver_data, height, width)

#使用确定的模型来校正图像。
# Load coefficients from previous calculation if need to
# (xcenter, ycenter, list_fact) = io.load_metadata_txt(
#     output_base + "/coefficients_radial_distortion.txt")
# Correct the image
corrected_mat = post.unwarp_image_backward(mat0, xcenter, ycenter, list_fact)
# Save results. Note that the output is 32-bit numpy array. Convert to lower-bit if need to.
io.save_image(output_base + "/corrected_image.tif", corrected_mat)
io.save_image(output_base + "/difference.tif", corrected_mat - mat0)