import os
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from numpy import random
from random import randint

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import skimage
from sort import *  # SORT 모듈 임포팅: SORT 트래킹


# ============================= 기존 YOLOv7 코드 커스텀: 추적기 메소드 =============================
""" 절대 픽셀 값에서 상대 바운딩 박스 계산 """
def bounding_box_rel(*xyxy):
    bounding_box_left = min([xyxy[0].item(), xyxy[2].item()])
    bounding_box_top = min([xyxy[1].item(), xyxy[3].item()])
    bounding_box_width = abs(xyxy[0].item() - xyxy[2].item())
    bounding_box_height = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bounding_box_left + bounding_box_width / 2)
    y_c = (bounding_box_top + bounding_box_height / 2)
    w = bounding_box_width
    h = bounding_box_height
    return x_c, y_c, w, h


""" 바운딩 박스 그리기 """
def draw_boxes(img, bounding_box, identities=None, categories=None, confidences=None, names=None, colors=None, offset=(0, 0)):
    for i, box in enumerate(bounding_box):
        x1, y1, x2, y2 = [int(i) for i in box]

        object_category = int(categories[i]) if categories is not None else 0
        object_identity = int(identities[i]) if identities is not None else 0
        # object_confidence = confidences[i] if confidences is not None else 0
        color = colors[object_category]

        tl = opt.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        if not opt.nobbox:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        if not opt.nolabel:
            label = str(object_identity) + ":" + names[object_category] if identities is not None else f'{names[object_category]} {confidences[i]:.2f}'
            tf = max(tl - 1, 1)  # 폰트 두께(font thickness)
            text_size = cv2.getTextSize(label, 0, fontScale=(tl / 3), thickness=tf)[0]
            c2 = x1 + text_size[0], y1 - text_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return img
# ============================= 추적기 메소드 끝 ============================= """


""" SORT 초기화 """
def detect(save_img=False):
    source, weights, view_img, save_txt, image_size, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # ================================ 기존 YOLOv7 코드 커스텀: 추적기 시각화를 위한 변수 선언 ================================
    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
    # ================================ 추적기 시각화를 위한 변수 선언 끝 ================================

    # ============================= 기존 YOLOv7 코드 커스텀: 디렉토리 생성 =============================
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    if not opt.nosave:
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # <============================= 디렉토리 생성 끝 =============================

    # 초기화
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # 모델 로드
    model = attempt_load(weights, map_location=device)  # FP32 모델 로드
    stride = int(model.stride.max())  # 가중치 업데이트시 사용되는 스트라이드를 모델의 최대 스트라이드로 설정
    image_size = check_img_size(image_size, s=stride)  # 이미지 사이즈 체크

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False

    if classify:
        model_classifier = load_classifier(name='resnet101', n=2)  # initialize
        model_classifier.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # 데이터 로더 설정
    vid_path, vid_writer = None, None

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=image_size, stride=stride)
    else:
        dataset = LoadImages(source, img_size=image_size, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, image_size, image_size).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = image_size
    old_img_b = 1  # 배치 사이즈

    t0 = time.time()

    # ================================ 기존 YOLOv7 코드 커스텀: 시작 시각 저장 변수 선언 ================================
    startTime = 0
    # ================================ 시작 시각 저장 변수 선언 끝 ================================

    """ 기존 YOLOv7 코드 """
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 --> fp16/32. fp: 부동 소수점(floating point)
        img /= 255.0  # 0 ~ 255 --> 0.0 ~ 1.0로 정규화

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]

            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS(Non-maximum suppression; 최대 억제)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, model_classifier, img, im0s)

        # Process detections
        # pred = model(img, augment=opt.augment)[0]
        for i, detection in enumerate(pred):  # detections per image
            if webcam:  # 배치 사이즈 >= 1 인 경우
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(detection):
                # Rescale boxes from img_size to im0 size
                detection[:, :4] = scale_coords(img.shape[2:], detection[:, :4], im0.shape).round()

                # Print results
                for c in detection[:, -1].unique():
                    n = (detection[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # ================================ 기존 YOLOv7 코드 커스텀: 추적기 메서드 사용 ================================
                # sort에 빈 어레이 전달
                detections_to_sort = np.empty((0, 6))

                # 참고: 감지된 오브젝트 클래스도 전달
                for x1, y1, x2, y2, conf, detection_class in detection.cpu().detach().numpy():
                    detections_to_sort = np.vstack((detections_to_sort, np.array([x1, y1, x2, y2, conf, detection_class])))

                # sort 메서드 호출
                tracked_detections = sort_tracker.update(detections_to_sort, opt.unique_track_color)
                tracks = sort_tracker.getTrackers()

                for t, track in enumerate(tracks):
                    track_color = colors[int(track.detclass)] if not opt.unique_track_color else sort_tracker.color_list[t]

                    # 트래킹 라인 시각화
                    [cv2.line(im0,
                              (int(track.centroidarr[i][0]), int(track.centroidarr[i][1])),
                              (int(track.centroidarr[i + 1][0]), int(track.centroidarr[i + 1][1])),
                              track_color,
                              thickness=opt.thickness)
                     for i, _ in enumerate(track.centroidarr)
                     if i < len(track.centroidarr) - 1]

                # 박스 표시(시각화)
                if len(tracked_detections) > 0:
                    bbox_xyxy = tracked_detections[:, :4]
                    identities = tracked_detections[:, 8]
                    categories = tracked_detections[:, 4]
                    confidences = None
                else:
                    bbox_xyxy = detections_to_sort[:, :4]
                    identities = None
                    categories = detections_to_sort[:, 5]
                    confidences = detections_to_sort[:, 4]

                im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors)
                # ================================ 추적기 메서드 사용 끝 ================================

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # ================================ 기존 YOLOv7 코드 커스텀: FPS 계산 코드 추가 ================================
            if dataset.mode != 'image' and opt.show_fps:
                currentTime = time.time()

                fps = 1 / (currentTime - startTime)
                startTime = currentTime

                cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            # ================================ FPS 계산 코드 추가 끝 ================================

            """ 기존 YOLOv7 코드 """
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 스트리밍 프로세스 종료: 그냥 컨트롤 + C 하면 강제 종료
                    cv2.destroyAllWindows()
                    raise StopIteration

            # 결과 저정(디텍션 + 이미지)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 비디오 또는 스트리밍
                    if vid_path != save_path:  # 새 비디오
                        vid_path = save_path

                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        if vid_cap:  # 비디오
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # 스트리밍
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels were saved to {save_dir / 'labels'}." if save_txt else ''
        print(f"Results were saved to {save_dir}{s}.")

    print(f'Process now done in ({time.time() - t0:.3f} seconds.)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    parser.add_argument('--track', action='store_true', help='run tracking')
    parser.add_argument('--show-track', action='store_true', help='show tracked path')
    parser.add_argument('--show-fps', action='store_true', help='show fps')
    parser.add_argument('--thickness', type=int, default=2, help='bounding box and font size thickness')
    parser.add_argument('--seed', type=int, default=1, help='random seed to control bbox colors')
    parser.add_argument('--nobbox', action='store_true', help='don`t show bounding box')
    parser.add_argument('--nolabel', action='store_true', help='don`t show label')
    parser.add_argument('--unique-track-color', action='store_true', help='show each track in unique color')

    opt = parser.parse_args()

    print(opt)

    np.random.seed(opt.seed)

    sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)

    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
