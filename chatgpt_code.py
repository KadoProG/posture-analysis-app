import cv2
import numpy as np

protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose/mpi/pose_iter_160000.caffemodel"

POSE_PAIRS = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7],
    [1, 14],
    [14, 8],
    [8, 9],
    [9, 10],
    [14, 11],
    [11, 12],
    [12, 13],
]

inWidth = 368
inHeight = 368


# OpenPoseの初期化
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# カメラの初期化
cap = cv2.VideoCapture(0)

while True:
    # フレームの取得
    ret, frame = cap.read()

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    # OpenPoseを使用して骨格情報を抽出
    blob = cv2.dnn.blobFromImage(
        frame,
        1.0 / 255.0,
        (inWidth, inHeight),
        (127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]

    # 骨格情報を元に映像に描画
    for i in range(output.shape[1]):
        points = []
        # print(str(output.shape[1]) + " " + str(output.shape[2]))
        for j in range(output.shape[2]):
            prob_map = output[0, i, j, 2]
            # print(prob_map)
            if prob_map > 0.1:  # 信頼度の閾値
                x = int(output[0, i, j, 0] * frame.shape[1])
                y = int(output[0, i, j, 1] * frame.shape[0])
                points.append((x, y))

        # 骨格を線で結ぶ
        for k in range(len(points) - 1):
            cv2.line(frame, points[k], points[k + 1], (0, 255, 0), 2)

    for i in range(15):
        probMap = output[0, i, :, :]
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    if prob > 0.1:
        cv2.circle(
            frame,
            (int(x), int(y)),
            8,
            (0, 255, 255),
            thickness=-1,
            lineType=cv2.FILLED,
        )
        cv2.putText(
            frame,
            "{}".format(i),
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            frame, (int(x), int(y)), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED
        )

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else:
        points.append(None)

    # 映像の表示
    cv2.imshow("Skeleton Tracking", frame)

    # 終了条件（'q'キーで終了）
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 後片付け
cap.release()
cv2.destroyAllWindows()
