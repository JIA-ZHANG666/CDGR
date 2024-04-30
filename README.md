# CDGR
CDGR:Cross-Modal Dual Graph Reasoning for Weakly Supervised Semantic Segmentation

Note:The external knowledge graph CM_kg_57_info.json obtained by ConceptNet. It contains the relationship matrix of PASCAL VOC 20 classes (20×20) and MSCOCO 80 classes (80×80).

Step 1. Prepare dataset.

Download PASCAL VOC 2012 devkit from official website.

Step 2. Train classification network and generate CAM seeds.

python run_sample.py --voc12_root ./VOCdevkit/VOC2012/ --work_space YOUR_WORK_SPACE --train_cam_pass True --make_cam_pass True --make_lpcam_pass True --eval_cam_pass True

Step 3. Train IRN and generate pseudo masks.

python run_sample.py --voc12_root ./VOCdevkit/VOC2012/ --work_space YOUR_WORK_SPACE --cam_to_ir_label_pass True --train_irn_pass True --make_sem_seg_pass True --eval_sem_seg_pass True

Step 4. Train semantic segmentation network.

To train DeepLab-v2, we refer to deeplab-pytorch. Please replace the groundtruth masks with generated pseudo masks.
