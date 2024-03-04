# Ref: https://www.kaggle.com/code/eduardtrulls/imc-2023-submission-example
'''
Baseline submission¶
A notebook to generate a valid submission. Implements three local feature/matcher methods: LoFTR, DISK, and KeyNetAffNetHardNet.

Remember to enable a GPU accelerator and disable internet access, then press "submit" on the right pane.
'''

# General utilities
import os
from tqdm import tqdm
from time import time
from fastprogress import progress_bar
import gc
import numpy as np
import h5py
from IPython.display import clear_output
from collections import defaultdict
from copy import deepcopy

# CV/ML
import cv2
import torch
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# 3D reconstruction
import pycolmap


# ------------------------------------------
print('Kornia version', K.__version__)
print('Pycolmap version', pycolmap.__version__)

LOCAL_FEATURE = 'KeyNetAffNetHardNet'
device=torch.device('cuda')
# Can be LoFTR, KeyNetAffNetHardNet, or DISK

def arr_to_str(a):
    return ';'.join([str(x) for x in a.reshape(-1)])

def load_torch_image(fname, device=torch.device('cpu')):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.
    img = K.color.bgr_to_rgb(img.to(device))
    return img

# ------------------------------------------
# We will use ViT global descriptor to get matching shortlists.
def get_global_desc(fnames, model,
                    device =  torch.device('cpu')):
    model = model.eval()
    model= model.to(device)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    global_descs_convnext=[]
    for i, img_fname_full in tqdm(enumerate(fnames),total= len(fnames)):
        key = os.path.splitext(os.path.basename(img_fname_full))[0]
        img = Image.open(img_fname_full).convert('RGB')
        timg = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            desc = model.forward_features(timg.to(device)).mean(dim=(-1,2))#
            #print (desc.shape)
            desc = desc.view(1, -1)
            desc_norm = F.normalize(desc, dim=1, p=2)
        #print (desc_norm)
        global_descs_convnext.append(desc_norm.detach().cpu())
    global_descs_all = torch.cat(global_descs_convnext, dim=0)
    return global_descs_all


def get_img_pairs_exhaustive(img_fnames):
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(i+1, len(img_fnames)):
            index_pairs.append((i,j))
    return index_pairs


def get_image_pairs_shortlist(fnames,
                              sim_th = 0.6, # should be strict
                              min_pairs = 20,
                              exhaustive_if_less = 20,
                              device=torch.device('cpu')):
    num_imgs = len(fnames)

    if num_imgs <= exhaustive_if_less:
        return get_img_pairs_exhaustive(fnames)

    model = timm.create_model('tf_efficientnet_b7',
                              checkpoint_path='/kaggle/input/tf-efficientnet/pytorch/tf-efficientnet-b7/1/tf_efficientnet_b7_ra-6c08e654.pth')
    model.eval()
    descs = get_global_desc(fnames, model, device=device)
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    # removing half
    mask = dm <= sim_th
    total = 0
    matching_list = []
    ar = np.arange(num_imgs)
    already_there_set = []
    for st_idx in range(num_imgs-1):
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]  
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < 1000:
                matching_list.append(tuple(sorted((st_idx, idx.item()))))
                total+=1
    matching_list = sorted(list(set(matching_list)))
    return matching_list

# ------------------------------------------
# Code to manipulate a colmap database.
# Forked from https://github.com/colmap/colmap/blob/dev/scripts/python/database.py

# class COLMAPDatabase(sqlite3.Connection):

# ------------------------------------------
# Code to interface DISK with Colmap.
# Forked from https://github.com/cvlab-epfl/disk/blob/37f1f7e971cea3055bb5ccfc4cf28bfd643fa339/colmap/h5_to_db.py


# ------------------------------------------
# Making kornia local features loading w/o internet
# load weights from local file
class KeyNetAffNetHardNet(KF.LocalFeature):
    """Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor.

    .. image:: _static/img/keynet_affnet.jpg
    """

    def __init__(
        self,
        num_features: int = 5000,
        upright: bool = False,
        device = torch.device('cpu'),
        scale_laf: float = 1.0,
            ):
        ori_module = KF.PassLAF() if upright else KF.LAFOrienter(angle_detector=KF.OriNet(False)).eval()
        if not upright:
            weights = torch.load('/kaggle/input/kornia-local-feature-weights/OriNet.pth')['state_dict']
            ori_module.angle_detector.load_state_dict(weights)
        detector = KF.KeyNetDetector(
            False, num_features=num_features, ori_module=ori_module, aff_module=KF.LAFAffNetShapeEstimator(False).eval()
                ).to(device)
        kn_weights = torch.load('/kaggle/input/kornia-local-feature-weights/keynet_pytorch.pth')['state_dict']
        detector.model.load_state_dict(kn_weights)
        affnet_weights = torch.load('/kaggle/input/kornia-local-feature-weights/AffNet.pth')['state_dict']
        detector.aff.load_state_dict(affnet_weights)
        
        hardnet = KF.HardNet(False).eval()
        hn_weights = torch.load('/kaggle/input/kornia-local-feature-weights/HardNetLib.pth')['state_dict']
        hardnet.load_state_dict(hn_weights)
        descriptor = KF.LAFDescriptor(hardnet, patch_size=32, grayscale_descriptor=True).to(device)
        super().__init__(detector, descriptor, scale_laf)

# ------------------------------------------
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
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    with h5py.File(f'{feature_dir}/lafs.h5', mode='w') as f_laf, \
         h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
        for img_path in progress_bar(img_fnames):
            img_fname = img_path.split('/')[-1]
            key = img_fname
            with torch.inference_mode():
                timg = load_torch_image(img_path, device=device)
                H, W = timg.shape[2:]
                if resize_small_edge_to is None:
                    timg_resized = timg
                else:
                    timg_resized = K.geometry.resize(timg, resize_small_edge_to, antialias=True)
                    print(f'Resized {timg.shape} to {timg_resized.shape} (resize_small_edge_to={resize_small_edge_to})')
                h, w = timg_resized.shape[2:]
                if LOCAL_FEATURE == 'DISK':
                    features = disk(timg_resized, num_feats, pad_if_not_divisible=True)[0]
                    kps1, descs = features.keypoints, features.descriptors
                    
                    lafs = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
                if LOCAL_FEATURE == 'KeyNetAffNetHardNet':
                    lafs, resps, descs = feature(K.color.rgb_to_grayscale(timg_resized))
                lafs[:,:,0,:] *= float(W) / float(w)
                lafs[:,:,1,:] *= float(H) / float(h)
                desc_dim = descs.shape[-1]
                kpts = KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
                descs = descs.reshape(-1, desc_dim).detach().cpu().numpy()
                f_laf[key] = lafs.detach().cpu().numpy()
                f_kp[key] = kpts
                f_desc[key] = descs
    return

def get_unique_idxs(A, dim=0):
    # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
    unique, idx, counts = torch.unique(A, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0],device=cum_sum.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    return first_indices

def match_features(img_fnames,
                   index_pairs,
                   feature_dir = '.featureout',
                   device=torch.device('cpu'),
                   min_matches=15, 
                   force_mutual = True,
                   matching_alg='smnn'
                  ):
    assert matching_alg in ['smnn', 'adalam']
    with h5py.File(f'{feature_dir}/lafs.h5', mode='r') as f_laf, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
        h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:

        for pair_idx in progress_bar(index_pairs):
                    idx1, idx2 = pair_idx
                    fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
                    key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
                    lafs1 = torch.from_numpy(f_laf[key1][...]).to(device)
                    lafs2 = torch.from_numpy(f_laf[key2][...]).to(device)
                    desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
                    desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
                    if matching_alg == 'adalam':
                        img1, img2 = cv2.imread(fname1), cv2.imread(fname2)
                        hw1, hw2 = img1.shape[:2], img2.shape[:2]
                        adalam_config = KF.adalam.get_adalam_default_config()
                        #adalam_config['orientation_difference_threshold'] = None
                        #adalam_config['scale_rate_threshold'] = None
                        adalam_config['force_seed_mnn']= False
                        adalam_config['search_expansion'] = 16
                        adalam_config['ransac_iters'] = 128
                        adalam_config['device'] = device
                        dists, idxs = KF.match_adalam(desc1, desc2,
                                                      lafs1, lafs2, # Adalam takes into account also geometric information
                                                      hw1=hw1, hw2=hw2,
                                                      config=adalam_config) # Adalam also benefits from knowing image size
                    else:
                        dists, idxs = KF.match_smnn(desc1, desc2, 0.98)
                    if len(idxs)  == 0:
                        continue
                    # Force mutual nearest neighbors
                    if force_mutual:
                        first_indices = get_unique_idxs(idxs[:,1])
                        idxs = idxs[first_indices]
                        dists = dists[first_indices]
                    n_matches = len(idxs)
                    if False:
                        print (f'{key1}-{key2}: {n_matches} matches')
                    group  = f_match.require_group(key1)
                    if n_matches >= min_matches:
                         group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
    return

def match_loftr(img_fnames,
                   index_pairs,
                   feature_dir = '.featureout_loftr',
                   device=torch.device('cpu'),
                   min_matches=15, resize_to_ = (640, 480)):
    matcher = KF.LoFTR(pretrained=None)
    matcher.load_state_dict(torch.load('/kaggle/input/loftr/pytorch/outdoor/1/loftr_outdoor.ckpt')['state_dict'])
    matcher = matcher.to(device).eval()

    # First we do pairwise matching, and then extract "keypoints" from loftr matches.
    with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='w') as f_match:
        for pair_idx in progress_bar(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            # Load img1
            timg1 = K.color.rgb_to_grayscale(load_torch_image(fname1, device=device))
            H1, W1 = timg1.shape[2:]
            if H1 < W1:
                resize_to = resize_to_[1], resize_to_[0]
            else:
                resize_to = resize_to_
            timg_resized1 = K.geometry.resize(timg1, resize_to, antialias=True)
            h1, w1 = timg_resized1.shape[2:]

            # Load img2
            timg2 = K.color.rgb_to_grayscale(load_torch_image(fname2, device=device))
            H2, W2 = timg2.shape[2:]
            if H2 < W2:
                resize_to2 = resize_to[1], resize_to[0]
            else:
                resize_to2 = resize_to_
            timg_resized2 = K.geometry.resize(timg2, resize_to2, antialias=True)
            h2, w2 = timg_resized2.shape[2:]
            with torch.inference_mode():
                input_dict = {"image0": timg_resized1,"image1": timg_resized2}
                correspondences = matcher(input_dict)
            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()

            mkpts0[:,0] *= float(W1) / float(w1)
            mkpts0[:,1] *= float(H1) / float(h1)

            mkpts1[:,0] *= float(W2) / float(w2)
            mkpts1[:,1] *= float(H2) / float(h2)

            n_matches = len(mkpts1)
            group  = f_match.require_group(key1)
            if n_matches >= min_matches:
                 group.create_dataset(key2, data=np.concatenate([mkpts0, mkpts1], axis=1))

    # Let's find unique loftr pixels and group them together.
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts=defaultdict(int)
    with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='r') as f_match:
        for k1 in f_match.keys():
            group  = f_match[k1]
            for k2 in group.keys():
                matches = group[k2][...]
                total_kpts[k1]
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0]+=total_kpts[k1]
                current_match[:, 1]+=total_kpts[k2]
                total_kpts[k1]+=len(matches)
                total_kpts[k2]+=len(matches)
                match_indexes[k1][k2]=current_match

    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]),dim=0, return_inverse=True)
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:,0] = unique_match_idxs[k1][m2[:,0]]
            m2[:,1] = unique_match_idxs[k2][m2[:,1]]
            mkpts = np.concatenate([unique_kpts[k1][ m2[:,0]],
                                    unique_kpts[k2][  m2[:,1]],
                                   ],
                                   axis=1)
            unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            out_match[k1][k2] = m2_semiclean2.numpy()
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp:
        for k, kpts1 in unique_kpts.items():
            f_kp[k] = kpts1
    
    with h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        for k1, gr in out_match.items():
            group  = f_match.require_group(k1)
            for k2, match in gr.items():
                group[k2] = match
    return

def import_into_colmap(img_dir,
                       feature_dir ='.featureout',
                       database_path = 'colmap.db',
                       img_ext='.jpg'):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, img_dir, img_ext, 'simple-radial', single_camera)
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )

    db.commit()
    return

# ------------------------------------------
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

for dataset in data_dict:
    for scene in data_dict[dataset]:
        print(f'{dataset} / {scene} -> {len(data_dict[dataset][scene])} images')

out_results = {}
timings = {"shortlisting":[],
           "feature_detection": [],
           "feature_matching":[],
           "RANSAC": [],
           "Reconstruction": []}

# ------------------------------------------
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

# ------------------------------------------
gc.collect()
datasets = []
for dataset in data_dict:
    datasets.append(dataset)

for dataset in datasets:
    print(dataset)
    if dataset not in out_results:
        out_results[dataset] = {}
    for scene in data_dict[dataset]:
        print(scene)
        # Fail gently if the notebook has not been submitted and the test data is not populated.
        # You may want to run this on the training data in that case?
        img_dir = f'{src}/test/{dataset}/{scene}/images'
        if not os.path.exists(img_dir):
            continue
        # Wrap the meaty part in a try-except block.
        try:
            out_results[dataset][scene] = {}
            img_fnames = [f'{src}/test/{x}' for x in data_dict[dataset][scene]]
            print (f"Got {len(img_fnames)} images")
            feature_dir = f'featureout/{dataset}_{scene}'
            if not os.path.isdir(feature_dir):
                os.makedirs(feature_dir, exist_ok=True)
            t=time()
            index_pairs = get_image_pairs_shortlist(img_fnames,
                                  sim_th = 0.5, # should be strict
                                  min_pairs = 20, # we select at least min_pairs PER IMAGE with biggest similarity
                                  exhaustive_if_less = 20,
                                  device=device)
            t=time() -t 
            timings['shortlisting'].append(t)
            print (f'{len(index_pairs)}, pairs to match, {t:.4f} sec')
            gc.collect()
            t=time()
            if LOCAL_FEATURE != 'LoFTR':
                detect_features(img_fnames, 
                                2048,
                                feature_dir=feature_dir,
                                upright=True,
                                device=device,
                                resize_small_edge_to=600
                               )
                gc.collect()
                t=time() -t 
                timings['feature_detection'].append(t)
                print(f'Features detected in  {t:.4f} sec')
                t=time()
                match_features(img_fnames, index_pairs, feature_dir=feature_dir,device=device)
            else:
                match_loftr(img_fnames, index_pairs, feature_dir=feature_dir, device=device, resize_to_=(600, 800))
            t=time() -t 
            timings['feature_matching'].append(t)
            print(f'Features matched in  {t:.4f} sec')
            database_path = f'{feature_dir}/colmap.db'
            if os.path.isfile(database_path):
                os.remove(database_path)
            gc.collect()
            import_into_colmap(img_dir, feature_dir=feature_dir,database_path=database_path)
            output_path = f'{feature_dir}/colmap_rec_{LOCAL_FEATURE}'

            t=time()
            pycolmap.match_exhaustive(database_path)
            t=time() - t 
            timings['RANSAC'].append(t)
            print(f'RANSAC in  {t:.4f} sec')

            t=time()
            # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
            mapper_options = pycolmap.IncrementalMapperOptions()
            mapper_options.min_model_size = 3
            os.makedirs(output_path, exist_ok=True)
            maps = pycolmap.incremental_mapping(database_path=database_path, image_path=img_dir, output_path=output_path, options=mapper_options)
            print(maps)
            #clear_output(wait=False)
            t=time() - t
            timings['Reconstruction'].append(t)
            print(f'Reconstruction done in  {t:.4f} sec')
            imgs_registered  = 0
            best_idx = None
            print ("Looking for the best reconstruction")
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

