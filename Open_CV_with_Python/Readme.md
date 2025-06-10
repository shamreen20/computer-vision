# Open_CV_With_Python

This repository provides a comprehensive guide to using OpenCV with Python, covering essential image and video processing techniques. Each topic is organized in sub-folders with example code, explanations and outputs.

### Topics

##### Reading and Writing Images
- Load images from files using OpenCV's imread function in various formats.
- Save modified images to disk with imwrite for further use.
- Link to Folder/script: [Reading and Writing Images](./Reading_and_Writing_images/reading_and_writing.ipynb)



##### Working with Video Files
- Read and process video files frame-by-frame using OpenCV's VideoCapture.
- Write processed frames to a new video file with VideoWriter.
- Link to Folder/script: [Working with Video Files](./Working_with_video_files/working_with_video_files.ipynb)



##### Exploring Color Space
- Convert images between color spaces like RGB, HSV, and Grayscale.
- Analyze color space properties for tasks like segmentation and filtering.
- Link to Folder/script: [Exploring Color Space](./Exploring_Color_Space/exploring_color_models.ipynb)



##### Color Thresholding
- Segment images by applying thresholds to specific color channels or ranges.
- Useful for isolating objects based on color in various color spaces.
- Link to Folder/script: [Color Thresholding](./Color_Thresholding/color_thresholding.ipynb)



##### Image Resizing, Scaling, and Interpolation
- Resize images to different dimensions using OpenCV's resize function.
- Apply interpolation methods like nearest-neighbor or bilinear for quality control.
- Link to Folder/script: [Image Resizing, Scaling, and Interpolation](./Image_Resizing_Scaling_and_interpolation/resizing_scaling_interpolation.ipynb)



##### Flip, Rotate, and Crop Images
- Flip images horizontally or vertically and rotate them by specific angles.
- Crop regions of interest to focus on specific parts of an image.
- Link to Folder/script: [Flip, Rotate, and Crop Images](./Flip_Rotate_and_Crop_Images/Flip_Rotate_Crop.ipynb)



##### Drawing Lines and Shapes Using OpenCV
- Draw lines, rectangles, circles, and polygons on images with OpenCV functions.
- Customize shapes with colors, thickness, and fill options for annotations.
- Link to Folder/script: [Drawing Lines and Shapes Using OpenCV](./Drawing_lines_and_shapes_using_opencv/draw_lines_shapes.ipynb)



##### Adding Text to Images
- Overlay text on images using OpenCV's putText with customizable fonts.
- Control text properties like size, color, and position for labeling.
- Link to Folder/script: [Adding Text to Images](./Adding_Text_to_images/text_opencv.ipynb)



##### Affine and Perspective Transformation
- Apply affine transformations to translate, rotate, or scale images geometrically.
- Use perspective transformations to correct distortions or change viewpoints.
- Link to Folder/script: [Affine and Perspective Transformation](./Affine_and_Perspective_Transformation/affine_perspective.ipynb)



##### Image Filters
- Apply various filters to preprocess images for noise reduction or enhancement.
- Explore convolution-based techniques for custom image processing tasks.
- Link to Folder/script: [Image Filters](./Image_FIlters/image_filters.ipynb)



##### Applying Blur Filter: Average, Gaussian, Median
- Smooth images using average, Gaussian, or median blur to reduce noise.
- Each method offers unique advantages for different image processing needs.
- Link to Folder/script: [Applying Blur Filter: Average, Gaussian, Median](./Applying_Blur_filter_Average_Gaussian_Median/blur.ipynb)



##### Edge Detection Using Sobel, Canny, & Laplacian
- Detect edges using Sobel for gradient-based edges, Canny for precise boundaries, or Laplacian for second-order derivatives.
- Essential for feature extraction and object detection in images.
- Link to Folder/script: [Edge Detection Using Sobel, Canny, & Laplacian](./Edge_Detection_Using_Sobel_Canny_&_Laplacian/edge_detection.ipynb)



##### Calculating and Plotting Histograms
- Compute histograms to analyze the intensity distribution of image pixels.
- Visualize histograms to understand image contrast and brightness properties.
- Link to Folder/script: [Calculating and Plotting Histograms](./Calculating_and_Plotting_Histograms/histogram.ipynb)



##### Histogram Equalization
- Enhance image contrast by redistributing pixel intensities using histogram equalization.
- Improves visibility in low-contrast images for better analysis.
- Link to Folder/script: [Histogram Equalization](./Histogram_Equalization/hist_equalization.ipynb)



##### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Apply CLAHE to enhance contrast locally while limiting noise amplification.
- Ideal for improving details in specific image regions without overexposure.
- Link to Folder/script: [CLAHE](./CLAHE/clahe.ipynb)



##### Contours
- Detect and extract contours to identify object boundaries in images.
- Use contour properties for shape analysis and object recognition tasks.
- Link to Folder/script: [Contours](./Contours/contour.ipynbcon)



##### Image Segmentation Using OpenCV
- Segment images into meaningful regions using techniques like thresholding or clustering.
- Useful for isolating objects or regions for further processing.
- Link to Folder/script: [Image Segmentation Using OpenCV](./Image_Segmentation_Using_openCV/image_segmentation.ipynb)



##### Haar Cascade for Face Detection
- Detect faces in images or videos using pre-trained Haar Cascade classifiers.
- Efficient for real-time face detection in various applications.
- Link to Folder/script: [Haar Cascade for Face Detection](./Haar_Cascade_for_face_detection/haar-cascade.ipynb)

### Getting Started

Install OpenCV using ````pip install opencv-python````. Each sub-folder contains Python scripts and documentation for hands-on learning of the respective topic.

### Contact
For questions or suggestions, please open an issue on the GitHub repository or contact [shamreen.tabassum@mailbox.tu-dresden.de].