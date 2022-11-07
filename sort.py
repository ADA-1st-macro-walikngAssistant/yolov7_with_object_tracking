from __future__ import print_function

import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from random import randint
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter


def get_color():
    # r = randint(0, 255)
    # g = randint(0, 255)
    # b = randint(0, 255)
    color = (randint(0, 255), randint(0, 255), randint(0, 255))
    return color


def linear_assignment(cost_matrix):
    try:
        import lap  # 선형 할당 문제 해결용
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])

    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


# [x1, y1, x2, y2] 형식의 두 박스 간의 IOU(intersection over union) 계산 메서드
def iou_batch(bb_test, bb_gt):
    # bb_gt: 바운딩 박스 실측값(ground truth)
    # bb_test: 바운딩 박스 테스트값
    # .expand_dims(a, b): 차원 추가 메서드. a: 추가할 차원의 위치, b: 추가할 차원의 크기
    # 참고: https://i.ytimg.com/vi/cwzGiiX59aw/maxresdefault.jpg
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    # xx1: 두 바운딩 박스의 x 좌표 중 큰 값
    # yy1: 두 바운딩 박스의 y 좌표 중 큰 값
    # xx2: 두 바운딩 박스의 x 좌표 중 작은 값
    # yy2: 두 바운딩 박스의 y 좌표 중 작은 값
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)  # width
    h = np.maximum(0., yy2 - yy1)  # height
    wh = w * h  # area

    # 중첩 영역 계산
    # o: 중첩 영역
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
              - wh)
    return o


# [x1, y1, x2, y2] 형식의 바운딩 박스를 받아 와서 [x, y, s, r] 형식의 z를 반환
# 참고: 여기서 x,y는 상자의 중심(center), s는 축척/영역(scale/area), r은 종횡비(aspect ratio)를 나타냄
def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)  # float(h): division by zero 방지
    return np.array([x, y, s, r]).reshape((4, 1))


# [x, y, s, r] 형식의 바운딩 박스를 받아 와서 [x1, y1, x2, y2] 형식으로 반환
# 참고: 여기서 x1, y1은 좌상단 모서리 좌표를, x2, y2는 우하단 모서리 좌표를 나타냄
def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w

    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


# 칼만 박스 트래커 클래스는 바운딩 박스로 관찰된, 개별적으로 추적되는 개체의 내부 상태를 나타냄
# 참고: 칼만 필터 https://en.wikipedia.org/wiki/Kalman_filter
#             https://ratsgo.github.io/statistics/2017/09/06/kalman/
#             https://ratsgo.github.io/machine%20learning/2017/10/09/Kalman/
class KalmanBoxTracker(object):
    count = 0  # 추적된 객체의 수

    # 트래커 생성: 이니셜 바운딩 박스 사용
    # 매개 변수 'bbox'는 -1 위치에 'detected class' 정수 넘버가 있어야 함.
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)  # 상태 벡터(state vector) x의 크기는 7, 관측 벡터 z의 크기는 4

        """
        self.x = zeros((dim_x, 1))                # state
        self.P = eye(dim_x)                       # uncertainty covariance
        self.Q = eye(dim_x)                       # process uncertainty
        self.B = None                             # control transition matrix
        self.F = eye(dim_x)                       # state transition matrix
        self.H = zeros((dim_z, dim_x))            # Measurement function
        self.R = eye(dim_z)                       # state uncertainty
        self._alpha_sq = 1.                       # fading memory control
        self.M = np.zeros((dim_z, dim_z))         # process-measurement cross correlation
        self.z = np.array([[None]*self.dim_z]).T  # last measurement
        self.K = np.zeros((dim_x, dim_z))         # kalman gain
        self.y = zeros((dim_z, 1))                # residual
        self.S = np.zeros((dim_z, dim_z))         # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))        # inverse system uncertainty
        """
        self.kf.F = np.array(  # F: 상태 전이 행렬(state transition matrix)
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1]
            ]
        )

        self.kf.H = np.array(  # H: 관측 행렬(observation matrix)
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0]
            ]
        )

        self.kf.R[2:, 2:] *= 10.  # R: 측정된 노이즈에 대한 공분산 행렬. 노이즈 인풋에 높은 값을 설정하면 박스에 더 많은 관성을 지정
        self.kf.P[4:, 4:] *= 1000.  # 관찰 불가능한 초기 속도에 대해 높은 불확실성 부여
        self.kf.P *= 10.

        # Q: 프로세스 노이즈의 공분산 행렬. 비정상적으로 움직이는 사물에 대해 높음으로 설정. 이 값은 추적기의 신뢰도를 결정함
        self.kf.Q[-1, -1] *= 0.5  # [-1, -1]: 속도에 대한 노이즈
        self.kf.Q[4:, 4:] *= 0.5  # [4:, 4:]: 위치에 대한 노이즈

        self.kf.x[:4] = convert_bbox_to_z(bbox)  # 상태 벡터(state vector) x의 초기값 설정
        self.time_since_update = 0  # 마지막 관측 이후 경과된 프레임 수
        self.id = KalmanBoxTracker.count  # 트래커 ID
        KalmanBoxTracker.count += 1  # 트래커 ID 증가

        self.history = []  # 추적기의 히스토리 저장
        self.hits = 0  # 추적기의 히트 수
        self.hit_streak = 0  # 연속 히트 수
        self.age = 0  # 추적기의 연령
        self.centroidarr = []  # 추적기의 중심점 저장
        CX = (bbox[0] + bbox[2]) // 2  # 중심점 x좌표
        CY = (bbox[1] + bbox[3]) // 2  # 중심점 y좌표
        self.centroidarr.append((CX, CY))  # centroidarr 리스트에 중심점 좌표 추가

        # YOLOv5 모델의 감지된 클래스 정보 유지
        self.detclass = bbox[5]

    def update(self, bbox):

        # 관측한 바운딩 박스로 추정된 상태 벡터(state vector) x 업데이트
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.detclass = bbox[5]  # detclass: YOLOv5 모델의 감지된 클래스 정보
        CX = (bbox[0] + bbox[2]) // 2
        CY = (bbox[1] + bbox[3]) // 2
        self.centroidarr.append((CX, CY))

    def predict(self):
        # 상태 벡터(state vector)를 전진시키고 바운딩 박스 추정치 반환
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0  # 속도가 음수인 경우 0으로 설정

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        # bbox = self.history[-1]
        # CX = (bbox[0] + bbox[2]) / 2
        # CY = (bbox[1]  +bbox[3]) / 2
        # self.centroidarr.append((CX, CY))

        return self.history[-1]

    # 현재 바운딩 박스 추정치 반환
    def get_state(self):
        """
        테스트 코드

        arr1 = np.array([[1, 2, 3, 4]])
        arr2 = np.array([0])
        arr3 = np.expand_dims(arr2, 0)
        np.concatenate((arr1,arr3), axis=1)
        """
        arr_detclass = np.expand_dims(np.array([self.detclass]), 0)

        arr_u_dot = np.expand_dims(self.kf.x[4], 0)
        arr_v_dot = np.expand_dims(self.kf.x[5], 0)
        arr_s_dot = np.expand_dims(self.kf.x[6], 0)

        return np.concatenate((convert_x_to_bbox(self.kf.x), arr_detclass, arr_u_dot, arr_v_dot, arr_s_dot), axis=1)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    추적된 객체에 탐지 프로세스 할당(둘 다 바운딩 박스로 표시)

    다음 값을 반환함
    1. 매칭된 객체
    2. 매칭되지 않은 탐지
    3. 매칭되지 않은 추적기
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)

        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)

    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []

    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []

    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # IOU 값이 낮는 경우 걸러냄(filter)
    matches = []

    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):

        # SORT 파라미터 설명
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.color_list = []

    def getTrackers(self, ):
        return self.trackers

    def update(self, dets=np.empty((0, 6)), unique_color=False):
        """
        파라미터:
        'dets': [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...] 형식의 numpy 감지(detection) 배열

        참고:
        1. 프레임 감지가 없는 경우에도 이 메서드를 호출해야 함 (np.empty((0,5)) 전달)
        2. 마지막 열이 개체 ID(신뢰도의 대체값)인 배열과 비슷한 배열을 반환함
        3. 반환되는 개체 수는 제공된 개체 수와 다를 때도 있다.
        """
        self.frame_count += 1

        # 이미 존재하는 추적기(detector)로부터 예측된 위치를 얻음
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]

            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)

            if unique_color:
                self.color_list.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # 할당된 탐지를 사용하여 일치하는 추적기 업데이트
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # 탐지 결과와 일치하지 않는 경우 새 추적기 생성 및 초기화
        for i in unmatched_dets:
            trk = KalmanBoxTracker(np.hstack((dets[i, :], np.array([0]))))
            self.trackers.append(trk)

            if unique_color:
                self.color_list.append(get_color())

        i = len(self.trackers)

        for trk in reversed(self.trackers):
            d = trk.get_state()[0]

            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # MOT 벤치마크용으로 1을 더해서 양수 값으로 만들어 줌
                # MOT 벤치마크: Multiple Object Tracking 성능 벤치마크

            i -= 1

            # 죽은 트랙렛(tracklet) 제거
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                if unique_color:
                    self.color_list.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)

        return np.empty((0, 6))


def parse_args():
    # 입력된 인수(arguments) 파싱
    parser = argparse.ArgumentParser(description='SORT demo')

    parser.add_argument('--display', dest='display',
                        help='온라인 추적기 출력 표시(느림) [False]',
                        action='store_true')

    parser.add_argument("--seq_path",
                        help="탐지 경로",
                        type=str, default='data')

    parser.add_argument("--phase",
                        help="seq_path의 하위 디렉토리",
                        type=str, default='train')

    parser.add_argument("--max_age",
                        help="연결된 감지 없이 트랙을 유지하기 위한 최대 프레임 수",
                        type=int,
                        default=1)

    parser.add_argument("--min_hits",
                        help="트래킹 초기화 전에 연결된 최소 감지 수",
                        type=int,
                        default=3)

    parser.add_argument("--iou_threshold",
                        help="매칭을 위한 최소 IOU",
                        type=float,
                        default=0.3)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # 결과 표시를 위해 사용함

    if display:
        if not os.path.exists('mot_benchmark'):
            print(
                '\n\tERROR: MOT 벤치마크 링크 없음\n\n    MOT 벤치마크에 심볼릭 링크 추가\n    (https://motchallenge.net/data/2D_MOT_2015/#download). 예:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
        exit()

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

    if not os.path.exists('output'):
        os.makedirs('output')

    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')

    for seq_dets_fn in glob.glob(pattern):
        mot_tracker = Sort(max_age=args.max_age,
                           min_hits=args.min_hits,
                           iou_threshold=args.iou_threshold)  # SORT 추적기 인스턴스 생성

    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')

    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]

    with open(os.path.join('output', '%s.txt' % (seq)), 'w') as out_file:
        print("Processing %s." % (seq))

        for frame in range(int(seq_dets[:, 0].max())):
            frame += 1  # 탐지 넘버 및 프레임 넘버 --> 1에서 시작
            dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
            dets[:, 2:4] += dets[:, 0:2]  # [x1, y1, w, h] --> [x1, y1, x2, y2] 변환
            total_frames += 1

        if display:
            fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg' % (frame))
            im = io.imread(fn)
            ax1.imshow(im)
            plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)

            if display:
                d = d.astype(np.int32)
                ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3, ec=colours[d[4] % 32, :]))
                # lw: 선 굵기, ec: 선 색상

        if display:
            fig.canvas.flush_events()
            plt.draw()
            ax1.cla()

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

    if display:
        print("Note: to get real runtime results run without the option: --display")
