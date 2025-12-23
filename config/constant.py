import os
class zsz_Constants:
    BASE_HALF_PIXEL = 5000 #认为最中间的1万像素都是膜
    WIDTH = 80000 #基础检测宽度(um)
    MIN_BACKGROUND = 210 #最低背景值，低于的不看做是背景
    NOICE_GRAY = 20 #噪声点

    MIN_GRAY =40 #识别为缺陷的最小阈值
    MAX_GRAY = 60

    MIN_DARK = 105 #识别为黑色区域的最小阈值
    MAX_DARK = 125

    GRAY_VALUE = 60
    DARK_VALUE = 115
    
    DARK_RATIO = 0.2

    Normal_BACKGROUND = 220 #
    MIN_PIXELS = 8 #最少多少个点算作是缺陷

class Constants:
    MAX_BUFFER_IMAGE = 10   # 缓冲队列最大值
    MAX_BUFFER_DEFECT= 140
    MIN_AVG_CORRECTION = 210
    MAX_AVG_CORRECTION = 230
    DEFAULT_CCF = 3074
    UM_PER_PIXEL = 7.3188
    MAX_WORKERS = 4
    BATCH_SIZE = 6
    MAX_QUEUE_SIZE = 200
    CATEGORY = [
        "_25",
        "25-50",
        "50-100",
        "100-200",
        "200-300",
        "300-400",
        "400-500",
        "500-600",
        "600-700",
        "700-800",
        "O800"
    ]
    SIZE_RANGES = ["<25um","25-50um", "50-100um", "101-200um", "201-300um", "301-400um"]

                       

class Machine_Learning:
    UPPER_BOUND = 190
    DARK_BOUND = 95
    DARK_VALUE = 0.12
    SHAPE_VALUE = 0.2


class Path:
    # 使用相对于当前工作目录的路径
    OUTPUT_DIR = os.path.join(os.getcwd(), "output", "defect_save")
    IMAGE_SAVE_DIR = os.path.join(os.getcwd(), "output", "images_save")
    test_image_dir = os.path.join(os.getcwd(), "test_data", "16384_4015")
    REPORT_DIR = os.path.join(os.getcwd(), "report_file")
    CCF_DIR = os.path.join(os.getcwd(), "camera", "config")
    CCF_PIXEL_DIR = os.path.join(os.getcwd(), "camera", "pixel")
    CAMERA_DIR = os.path.join(os.getcwd(), "camera", "x64", "Release", "camera_dll.dll")


class Border:
    N=3175
    M=12825


class Default:
    default_sample = "sample"
    default_user = "user"


class Channal:
    CH="A"
    Port="COM8"

class Light:
    NORMAL = 0   #正常