## IMC-2023 submission
- 特征提取匹配总体两种方式：
  - 分两步： 特征提取+匹配
  - 端到端匹配

- 提交按照固定模式
- 不联网（加载模型本地文件）
- 常用模型：SP+SG、LoFTR、KeyNetAffNetHardNet(KeyNet detector + AffNet + HardNet descriptor)、hloc、pixel-perfect-sfm、adalam、NN-ratio、NN-mutual、R2D2、SIFT、DISK

## submission structure
Ref: https://www.kaggle.com/code/eduardtrulls/imc-2023-submission-example
>Implements three local feature/matcher methods: LoFTR, DISK, and KeyNetAffNetHardNet.
Remember to enable a GPU accelerator and disable internet access, then press "submit" on the right pane.

提交的代码基本都使用的该文档的方式。

```py
class KeyNetAffNetHardNet(KF.LocalFeature):
    ...

def detect_features(img_fnames,
                    num_feats = 2048,
                    upright = False,
                    device=torch.device('cpu'),
                    feature_dir = '.featureout',
                    resize_small_edge_to = 600):
    if LOCAL_FEATURE == 'DISK':
        # Load DISK from Kaggle models so it can run when the notebook is offline.
        disk = KF.DISK().to(device)
        pretrained_dict = torch.load('/kaggle/input/disk/pytorch/depth-supervision/1/loftr_outdoor.ckpt', map_location=device)
        disk.load_state_dict(pretrained_dict['extractor'])
        disk.eval()
    if LOCAL_FEATURE == 'KeyNetAffNetHardNet':
        feature = KeyNetAffNetHardNet(num_feats, upright, device).to(device).eval()
    ...
    with h5py.File(f'{feature_dir}/lafs.h5', mode='w') as f_laf, \
         h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
        for img_path in progress_bar(img_fnames):
            ...
            with torch.inference_mode():
                ...
                if LOCAL_FEATURE == 'DISK':
                    features = disk(timg_resized, num_feats, pad_if_not_divisible=True)[0]
                    kps1, descs = features.keypoints, features.descriptors
                    
                    lafs = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
                if LOCAL_FEATURE == 'KeyNetAffNetHardNet':
                    ...
                ...

def match_features(img_fnames,
                   index_pairs,
                   feature_dir = '.featureout',
                   device=torch.device('cpu'),
                   min_matches=15, 
                   force_mutual = True,
                   matching_alg='smnn'
                  ):
    assert matching_alg in ['smnn', 'adalam']
    ...
    with h5py.File(f'{feature_dir}/lafs.h5', mode='r') as f_laf, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
        h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:

        for pair_idx in progress_bar(index_pairs):
                    ...
                    if matching_alg == 'adalam':
                        ...
                    else:
                        ...

def match_loftr(img_fnames,
                   index_pairs,
                   feature_dir = '.featureout_loftr',
                   device=torch.device('cpu'),
                   min_matches=15, resize_to_ = (640, 480)):
    matcher = KF.LoFTR(pretrained=None)
    matcher.load_state_dict(torch.load('/kaggle/input/loftr/pytorch/outdoor/1/loftr_outdoor.ckpt')['state_dict'])
    matcher = matcher.to(device).eval()

    with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='w') as f_match:
        for pair_idx in progress_bar(index_pairs):
            ... # preprocess: color-convert, resize
            with torch.inference_mode():
                input_dict = {"image0": timg_resized1,"image1": timg_resized2}
                correspondences = matcher(input_dict)
            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()

            mkpts0[:,0] *= float(W1) / float(w1)
            mkpts0[:,1] *= float(H1) / float(h1)
            ...
    # find unique loftr pixels and group them together.

# Function to create a submission file.
def create_submission(out_results, data_dict):
    with open(f'submission.csv', 'w') as f:
        f.write('image_path,dataset,scene,rotation_matrix,translation_vector\n')
        for dataset in data_dict:
            if dataset in out_results:
                res = out_results[dataset]
            else:
                res = {}
            for scene in data_dict[dataset]:
                if scene in res:
                    scene_res = res[scene]
                else:
                    scene_res = {"R":{}, "t":{}}
                for image in data_dict[dataset][scene]:
                    if image in scene_res:
                        print (image)
                        R = scene_res[image]['R'].reshape(-1)
                        T = scene_res[image]['t'].reshape(-1)
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    f.write(f'{image},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n')

## 提交
# Get data from csv.
data_dict = {}
with open(f'{src}/sample_submission.csv', 'r') as f:
    for i, l in enumerate(f):
        # Skip header.
        if l and i > 0:
            image, dataset, scene, _, _ = l.strip().split(',')
            if dataset not in data_dict:
                data_dict[dataset] = {}
            if scene not in data_dict[dataset]:
                data_dict[dataset][scene] = []
            data_dict[dataset][scene].append(image)       

out_results = {}
timings = {"shortlisting":[],
           "feature_detection": [],
           "feature_matching":[],
           "RANSAC": [],
           "Reconstruction": []}

for dataset in datasets:
    print(dataset)
    if dataset not in out_results:
        out_results[dataset] = {}
    for scene in data_dict[dataset]:        
        img_dir = f'{src}/test/{dataset}/{scene}/images'
        if not os.path.exists(img_dir):
            continue
        # Wrap the meaty part in a try-except block.
        try:
            feature_dir = f'featureout/{dataset}_{scene}'
            if not os.path.isdir(feature_dir):
                os.makedirs(feature_dir, exist_ok=True)
            index_pairs = get_image_pairs_shortlist(img_fnames,
                                  sim_th = 0.5, # should be strict
                                  min_pairs = 20, # we select at least min_pairs PER IMAGE with biggest similarity
                                  exhaustive_if_less = 20,
                                  device=device)
            ...
            if LOCAL_FEATURE != 'LoFTR':
                detect_features(img_fnames, 
                                2048,
                                feature_dir=feature_dir,
                                upright=True,
                                device=device,
                                resize_small_edge_to=600
                               )
                match_features(img_fnames, index_pairs, feature_dir=feature_dir,device=device)
            else:
                match_loftr(img_fnames, index_pairs, feature_dir=feature_dir, device=device, resize_to_=(600, 800))
            ...
            import_into_colmap(img_dir, feature_dir=feature_dir,database_path=database_path)
            output_path = f'{feature_dir}/colmap_rec_{LOCAL_FEATURE}'

            t=time()
            pycolmap.match_exhaustive(database_path)  # RANSAC
            t=time() - t 
            timings['RANSAC'].append(t)
            print(f'RANSAC in  {t:.4f} sec')

            t=time()
            # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
            mapper_options = pycolmap.IncrementalMapperOptions()
            mapper_options.min_model_size = 3
            os.makedirs(output_path, exist_ok=True)
            maps = pycolmap.incremental_mapping(database_path=database_path, image_path=img_dir, output_path=output_path, 
                                                options=mapper_options)
            t=time() - t
            timings['Reconstruction'].append(t)
            print(f'Reconstruction done in  {t:.4f} sec')
            
            print ("Looking for the best reconstruction")
            imgs_registered  = 0
            best_idx = None
            if isinstance(maps, dict):
                for idx1, rec in maps.items():
                    print (idx1, rec.summary())
                    if len(rec.images) > imgs_registered:
                        imgs_registered = len(rec.images)
                        best_idx = idx1
            if best_idx is not None:
                print (maps[best_idx].summary())
                for k, im in maps[best_idx].images.items():
                    key1 = f'{dataset}/{scene}/images/{im.name}'
                    out_results[dataset][scene][key1] = {}
                    out_results[dataset][scene][key1]["R"] = deepcopy(im.rotmat())
                    out_results[dataset][scene][key1]["t"] = deepcopy(np.array(im.tvec))
            print(f'Registered: {dataset} / {scene} -> {len(out_results[dataset][scene])} images')
            print(f'Total: {dataset} / {scene} -> {len(data_dict[dataset][scene])} images')
            create_submission(out_results, data_dict)
            gc.collect()
        except:
            pass

create_submission(out_results, data_dict)
```

## 重要参考
- https://www.kaggle.com/code/zhongwenhao/imc-2023-baseline-hloc 借助[hloc](https://github.com/cvg/Hierarchical-Localization/)实现SP+SG, LoFTR
- https://www.kaggle.com/code/maxchen303/imc2023-final-pub 包含获取基础矩阵、评估
- https://github.com/ubc-vision/image-matching-benchmark-baselines  local features