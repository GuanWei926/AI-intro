import os
import cv2
import glob
import numpy as np


def load_data_small():
    """
    This function loads images form the path: 'data/data_small' and return the training
    and testing dataset. The dataset is a list of tuples where the first element is the
    numpy array of shape (m, n) representing the image the second element is its
    classification (1 or 0).

    Parameters:
        None

    Returns:
        dataset: The first and second element represents the training and testing dataset respectively
    """

    # Begin your code (Part 1-1)
    dataset = []
    training_dataset = []
    testing_dataset = []
    data_path = "C:/Users/Microsoft/Desktop/HW1/HW1/data/data_small"
    for cate in os.listdir(data_path):
        cate_path = os.path.join(data_path, cate)  # concatenating the path(test or train)
        for lable in os.listdir(cate_path):
            lable_path = os.path.join(cate_path, lable)  # concatenating the path(face or non-face)
            for image_file in os.listdir(lable_path):
                image_path = os.path.join(lable_path, image_file)
                class_label = 1 if lable == "face" else 0
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image_info = np.array(image)
                if image is not None:
                    if cate == "train":
                        training_dataset.append((image_info, class_label))
                    else:
                        testing_dataset.append((image_info, class_label))
    dataset.append(training_dataset)
    dataset.append(testing_dataset)
    # End your code (Part 1-1)
    return dataset


def load_data_FDDB(data_idx=["01", "04"]):
    """
    This function generates the training and testing dataset  form the path: 'data/data_small'.
    The dataset is a list of tuples where the first element is the numpy array of shape (m, n)
    representing the image the second element is its classification (1 or 0).

    In the following, there are 4 main steps:
    1. Read the .txt file
    2. Crop the faces using the ground truth label in the .txt file
    3. Random crop the non-faces region
    4. Split the dataset into training dataset and testing dataset

    Parameters:
        data_idx: the data index string of the .txt file

    Returns:
        train_dataset: the training dataset
        test_dataset: the testing dataset
    """

    for idx in data_idx:
        with open(
            "C:/Users/Microsoft/Desktop/HW1/HW1/data/data_FDDB/FDDB-folds/FDDB-fold-{}-ellipseList.txt".format(
                idx
            )
        ) as file:
            line_list = [line.rstrip() for line in file]  # rstrip() removes the specified characters at the end of a string

        # Set random seed for reproducing same image croping results
        np.random.seed(0)

        face_dataset, nonface_dataset = [], []
        line_idx = 0

        # Iterate through the .txt file
        # The detail .txt file structure can be seen in the README at https://vis-www.cs.umass.edu/fddb/
        while line_idx < len(line_list):
            img_gray = cv2.imread(
                os.path.join("C:/Users/Microsoft/Desktop/HW1/HW1/data/data_FDDB",line_list[line_idx] + ".jpg",),
                cv2.IMREAD_GRAYSCALE,
            )
            num_faces = int(line_list[line_idx + 1])

            # Crop face region using the ground truth label
            face_box_list = []
            for i in range(num_faces):
                # Here, each face is denoted by:
                # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
                coord = [int(float(j)) for j in line_list[line_idx + 2 + i].split()]
                x, y = coord[3] - coord[1], coord[4] - coord[0]
                w, h = 2 * coord[1], 2 * coord[0]

                left_top = (max(x, 0), max(y, 0))
                right_bottom = (min(x + w, img_gray.shape[1]), min(y + h, img_gray.shape[0]),)
                face_box_list.append([left_top, right_bottom])
                # cv2.rectangle(img_gray, left_top, right_bottom, (0, 255, 0), 2)

                img_crop = img_gray[left_top[1] : right_bottom[1], left_top[0] : right_bottom[0]].copy()
                face_dataset.append((cv2.resize(img_crop, (19, 19)), 1))

            line_idx += num_faces + 2

            # Random crop N non-face region
            # Here we set N equal to the number of faces to generate a balanced dataset
            # Note that we have alreadly save the bounding box of faces into `face_box_list`, you can utilize it for non-face region cropping
            for i in range(num_faces):
                # Begin your code (Part 1-2)
                while True:
                    index = np.random.randint(len(face_box_list))
                    left_top, right_bottom = face_box_list[index][0], face_box_list[index][1]

                    # Randomly define the size of the non-face region (adjust as needed)
                    nonface_width = np.random.randint(1, 100)
                    nonface_height = np.random.randint(1, 100)

                    # Calculate the maximum allowed intersection area (%)
                    max_intersection = 0.1

                    # Randomly select a point for the top-left corner of the non-face region
                    x = np.random.randint(0, img_gray.shape[1] - nonface_width)
                    y = np.random.randint(0, img_gray.shape[0] - nonface_height)

                    # Calculate the intersection area with the selected face bounding box
                    intersection_area = 0
                    for index in range(len(face_box_list)):
                        left_top, right_bottom = face_box_list[index][0], face_box_list[index][1]
                        intersection_area += max(0, min(right_bottom[0], x + nonface_width) - max(left_top[0], x)) * \
                        max(0, min(right_bottom[1], y + nonface_height) - max(left_top[1], y))

                    # Check if the intersection is small enough
                    if intersection_area / (nonface_width * nonface_height) < max_intersection:
                        img_crop = img_gray[y : y + nonface_height, x : x + nonface_width].copy()
                        nonface_dataset.append((cv2.resize(img_crop, (19, 19)), 0))
                        break
                # End your code (Part 1-2)

                # nonface_dataset.append((cv2.resize(img_crop, (19, 19)), 0))

            # cv2.imshow("windows", img_gray)
            # cv2.waitKey(0)

    # train test split
    num_face_data, num_nonface_data = len(face_dataset), len(nonface_dataset)
    SPLIT_RATIO = 0.7

    train_dataset = (
        face_dataset[: int(SPLIT_RATIO * num_face_data)]
        + nonface_dataset[: int(SPLIT_RATIO * num_nonface_data)]
    )
    test_dataset = (
        face_dataset[int(SPLIT_RATIO * num_face_data) :]
        + nonface_dataset[int(SPLIT_RATIO * num_nonface_data) :]
    )

    return train_dataset, test_dataset


def create_dataset(data_type):
    if data_type == "small":
        return load_data_small()
    else:
        return load_data_FDDB()
