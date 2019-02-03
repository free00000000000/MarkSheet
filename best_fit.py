import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def partition(img_gray, W=120, H=220, threshold=0):
    print("Partitioning...")
    import time
    start_time = time.time()

    weigh = img_gray.shape[1]
    
    row = np.dot(img_gray, np.ones(weigh, np.int32))
    x = [i for i in range(row.size) if row[i] <= threshold]
    cut = []
    for i in range(len(x)-1):
        if not x[i]+1 == x[i+1]:
            if cut and x[i] - cut[-1][1] <= 10:  # 讓沒跟樂譜連在一起的符號不被切掉
                cut[-1][1] = x[i+1]
            else:
                cut.append([x[i], x[i+1]])
    
    # 找五線譜
    ''' 目前的方法不太好...'''
    hold = weigh*(2/3)*255
    tabs = []
    others = []
    for c in range(len(cut)):
        s = np.max(row[cut[c][0]:cut[c][1]])
        if s > hold:  # 是五線譜
            tabs.append(cut[c])
        else:
            others.append(cut[c])
    
    print("--- %s seconds ---" % (time.time() - start_time))

    # print(cut)

    # draw
    # img = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
    # for i in tabs:
    #     print(i)
    #     cv2.line(img, (0, i[0]), (weigh, i[0]), (150, 255, 255), 2)
    #     cv2.line(img, (0, i[1]), (weigh, i[1]), (150, 255, 255), 2)
    # cv2.imwrite('./output/partition.png', img)

    

    for i, y in enumerate(tabs):
        print(i, y)
        image = img_gray[y[0]:y[1]]
        # name = './' + str(i) + '.png'
        # cv2.imwrite(name, image)
        print(image.shape)
        h = image.shape[0]
        col = np.dot(image.transpose(), np.ones(h, np.int32))
        
        b = 0
        e = 0
        for x in range(1, len(col)):
            if col[x-1] == 0 and col[x] != 0:
                b = x
            elif col[x] == 0 and col[x-1] != 0:
                e = x
        print(b, e)
        X = []
        tabs_h = stats.mode(col).mode[0] + 5  # 誤差
        for x in range(b+W, e, W):
            if not col[x] <= tabs_h:
                k = 1
                while True:
                    if col[x+k] <= tabs_h:
                        x += k
                        break
                    elif col[x-k] <=tabs_h:
                        x -= k
                        break
                    k += 1
            
            X.append(x)
        
        print(X)

        X[-1] = e

        crop_img = img_gray[y[0]:y[0]+h, b:X[0]]
        cv2.imwrite('./0.png', crop_img)
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
        for j in range(1, len(X)):
            crop_img = img_gray[y[0]:y[0]+h, X[j-1]:X[j]]
            # crop_img = cv2.resize(crop_img, (120, 220))
            mm = './output/' + str(j) + '.png'
            cv2.imwrite(mm, crop_img)
            # cv2.imshow("cropped", crop_img)
            # cv2.waitKey(0)
        break


    return tabs, others



if __name__ == "__main__":
    img_file = "./resources/sheet.jpg"

    img = cv2.imread(img_file, 0)
    ret, img_gray = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    partition(img_gray)