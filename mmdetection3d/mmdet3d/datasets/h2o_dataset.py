from os import path as osp

import cv2
import mmcv

# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet3d.core import show_result
from mmdet3d.core.bbox import DepthInstance3DBoxes, LiDARInstance3DBoxes, get_box_type
from mmdet.datasets import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class H2ODataset(Custom3DDataset):
    CLASSES = (
        "dontcare",
        "book",
        "espresso",
        "lotion",
        "spray",
        "milk",
        "cocoa",
        "chips",
        "capuccino ",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refrigerator",
        "showercurtrain",
        "toilet",
        "sink",
        "bathtub",
        "garbagebin",
    )

    def __init__(
        self,
        data_root,
        ann_file,
        pipeline=None,
        classes=None,
        modality=None,
        box_type_3d="Depth",
        filter_empty_gt=True,
        test_mode=False,
        use_valid_flag=False,
        file_client_args=dict(backend="disk"),
    ):
        self.data_root = data_root
        self.ann_file = ann_file
        self.classes = classes
        self.modality = modality
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.filter_empty_gt = filter_empty_gt
        self.test_mode = test_mode
        self.use_valid_flag = use_valid_flag
        self.file_client = mmcv.FileClient(**file_client_args)

        # load annotations
        with self.file_client.get_local_path(self.ann_file) as local_path:
            self.data_infos = self.load_annotations(local_path)

        self.pipeline = Compose(pipeline)

        if not self.test_mode:
            self._set_group_flag()

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.list_from_file(ann_file, prefix=self.data_root)
        data_infos = [
            dict(
                idx=i,
                vid_path="/".join(fp.split("/")[:-2]),
                frm_name=fp.split("/")[-1].split(".")[0],
                img_prefix="/".join(fp.split("/")[:-1]),
            )
            for i, fp in enumerate(data[:100:20])
        ]
        for di in data_infos:
            di.update(
                dict(
                    pc_path=osp.join(di["vid_path"], "pc", f'{di["frm_name"]}.txt'),
                    depth=osp.join(di["vid_path"], "depth", f'{di["frm_name"]}.png'),
                    obj_meta=osp.join(di["vid_path"], "obj_pose", f'{di["frm_name"]}.txt'),
                )
            )
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - vid_filename (str): Filename of the video file.
                - img_prefix (str, optional): Path to images folder.
                - img_info (str, optional): Image metadata.
                - gt_bboxes (torch.Tensor): Ground truth bounding boxes
                - gt_labels (torch.Tensor): Ground truth object labels
                If in training mode:
                - ann_info (dict): Annotation info.
                - centers2d (torch.Tensor): Centers of bboxes.
                - depths (torch.Tensor): Depth parameter from dataset.
        """
        info = self.data_infos[index]
        camera_matrix = self._get_cam_matrix(info["vid_path"])
        img_info = dict(filename=f"{info['frm_name']}.png", cam_intrinsic=camera_matrix)
        input_dict = dict(
            sample_idx=info["idx"],
            pts_filename=info["pc_path"],
            vid_filename=info["vid_path"],
            img_prefix=info["img_prefix"],
            img_info=img_info,
            gt_bboxes=torch.zeros(4).unsqueeze(0),
            gt_labels=torch.zeros(0).unsqueeze(0),
        )

        if not self.test_mode:
            annos, centers, depth = self.get_ann_info(index, camera_matrix)
            input_dict.update(dict(ann_info=annos, centers2d=centers.long(), depths=depth))

        return input_dict

    def get_ann_info(self, index, cam_matrix):
        """Use index to get the annos, thus the evalhook could also use this
        api.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            - anns_results (dict): Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
            - gt_obj_centers (torch.Tensor): Centers of GT bboxes.
            - depth (torch.Tensor): Depth parameter from dataset.
        """
        info = self.data_infos[index]
        gt_labels_3d, gt_bboxes_3d, gt_obj_centers, depth = self._load_obj_data(
            info["obj_meta"], cam_matrix
        )
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1]
        ).convert_to(self.box_mode_3d)
        gt_labels_3d = np.array(gt_labels_3d)
        gt_names_3d = []
        for lid in gt_labels_3d:
            gt_names_3d.append(self.CLASSES[lid])

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, gt_names_3d=gt_names_3d
        )
        return anns_results, gt_obj_centers, depth

    def _load_obj_data(self, annotation_file, camera_matrix):
        with open(annotation_file) as f:
            all_lines = f.readlines()
            all_obj_pose = []
            all_obj_class = []
            obj_centers = []
            depth = []

            for lines in all_lines:
                lines = lines.strip().split()
                all_obj_class.append(int(float(lines[0])))
                obj_centers.append([float(c) for c in lines[1:4]])
                depth.append(int(obj_centers[-1][-1]))
                obj_pose = torch.Tensor([float(c) for c in lines[4 : (9 * 3 + 1)]])
                obj_pose = obj_pose.reshape(-1, 3)

                # Convert to MMDet bbox format
                xs = torch.norm(obj_pose[0] - obj_pose[3], 2)
                ys = torch.norm(obj_pose[0] - obj_pose[1], 2)
                zs = torch.norm(obj_pose[0] - obj_pose[4], 2)
                obj_pose = torch.Tensor(torch.cat([obj_pose[0, :], torch.Tensor([xs, ys, zs])]))
                all_obj_pose.append(obj_pose.unsqueeze(0))
            all_obj_pose = torch.cat(all_obj_pose)

            obj_centers, _ = cv2.projectPoints(
                np.array(obj_centers),
                np.array([[0, 0, 0]]).astype(np.float32),
                np.array([[0, 0, 0]]).astype(np.float32),
                camera_matrix,
                None,
            )
            obj_centers = torch.Tensor(obj_centers).squeeze(0)
            depth = torch.Tensor(depth)

        return all_obj_class, all_obj_pose, obj_centers, depth

    def _get_cam_matrix(self, fpath):
        intr_params = mmcv.list_from_file(osp.join(fpath, "cam_intrinsics.txt"))
        intr_params = intr_params[0].strip().split()
        cam_fx, cam_fy, cam_cx, cam_cy, _, _ = [float(p) for p in intr_params]
        cam_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
        return cam_mat

    def evaluate(
        self,
        results,
        metric=None,
        iou_thr=(0.25, 0.5),
        logger=None,
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Defaults to None.
            iou_thr (list[float]): AP IoU thresholds. Defaults to (0.25, 0.5).
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        from mmdet3d.core.evaluation import indoor_eval

        assert isinstance(results, list), f"Expect results to be list, got {type(results)}."
        assert len(results) > 0, "Expect length of results > 0."
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f"Expect elements in results to be dict, got {type(results[0])}."
        results = [i["img_bbox"] for i in results]

        for info in self.data_infos:
            camera_matrix = self._get_cam_matrix(info["vid_path"])
            obj_cls, obj_pose, _, _ = self._load_obj_data(info["obj_meta"], camera_matrix)
            info["annos"] = dict(gt_num=obj_pose.shape[0], gt_boxes_upright_depth=obj_pose)
            info["annos"]["class"] = obj_cls

        gt_annos = [info["annos"] for info in self.data_infos]
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
        ret_dict = indoor_eval(
            gt_annos,
            results,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d,
        )

        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return ret_dict
