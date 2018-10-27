import cv2
import numpy as np
import matplotlib.pyplot as plt

def partition(img, threshold=0):
    print("Partitioning...")
    import time
    start_time = time.time()

    weigh = img.shape[1]
    ret, img_gray = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    
    row = np.dot(img_gray, np.ones(weigh, np.int32))
    x = [i for i in range(row.size) if row[i] <= threshold]
    cut = []
    for i in range(len(x)-1):
        if not x[i]+1 == x[i+1]:
            if cut and x[i] - cut[-1][1] <= 10:
                cut[-1][1] = x[i+1]
            else:
                cut.append([x[i], x[i+1]])

    print("--- %s seconds ---" % (time.time() - start_time))

    # # draw
    # img = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
    # for i in cut:
    #     for j in i:
    #         cv2.line(img, (0, j), (weigh, j), (150, 255, 255), 2)
    # cv2.imwrite('./output/partition.png', img)

    return cut

if __name__ == "__main__":
    img_file = "./resources/samples/sheet.jpg"
    staff_file = "./resources/template/staff.png"
    # img_file = "../SheetVision/resources/samples/races.png"

    img = cv2.imread(img_file, 0)
    partition(img)